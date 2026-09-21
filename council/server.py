"""The web app.

Deployed on a Tailscale Funnel, which means anyone with the link can reach it.
Two consequences shape this file: the guards (size caps, a per-IP rate limit, a
ceiling on concurrent runs so one visitor cannot monopolise the machine's GPU),
and the storage policy — a submitted CV lives in memory for the duration of its
run and is deleted when the result is delivered. No submitted content touches
disk; the only file the service writes is the timing log (durations and counts,
never content — see council/timings.py).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from council import __version__
from council.config import settings
from council.extract import ExtractionError, from_upload
from council.personas import COUNCIL
from council.pipeline import run as run_council

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Hiring Council", version=__version__, docs_url=None, redoc_url=None)

# --- guards ---------------------------------------------------------------

_hits: dict[str, deque[float]] = defaultdict(deque)
_runs: dict[str, dict] = {}
_run_slots = asyncio.Semaphore(2)     # a laptop can honestly serve two at once


def _client(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= settings.rate_limit_per_hour:
        return True
    q.append(now)
    return False


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    )
    return resp


# --- pages ----------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/app.js")
async def appjs():
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


@app.get("/styles.css")
async def styles():
    return FileResponse(STATIC / "styles.css", media_type="text/css")


@app.get("/api/health")
async def health():
    from council.providers import Ollama

    models = await Ollama().available_models()
    return {
        "ok": bool(models),
        "version": __version__,
        "models_installed": models,
        "council_size": len(COUNCIL),
        "active_runs": len(_runs),
    }


@app.get("/api/council")
async def council_info():
    """The panel, its biases and its weights — published, not hidden."""
    return {
        "members": [
            {
                "name": m.name, "title": m.title, "bias": m.bias, "weights": m.weights,
                "offset": m.offset, "reliability": m.reliability, "reads": m.reads,
                "models": list(m.preferred_models[:2]),
            }
            for m in COUNCIL
        ]
    }


# --- the run --------------------------------------------------------------

async def _read_input(field: str, file: UploadFile | None, text: str, cap: int) -> tuple[str, str, dict]:
    """Returns (text, source, signals) — all three feed the ATS audit."""
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"{field} is larger than {settings.max_upload_bytes // (1024*1024)}MB.")
        try:
            body, source, signals = from_upload(file.filename, data)
        except ExtractionError as e:
            raise HTTPException(400, str(e))
        return body[:cap], source, signals
    return (text or "").strip()[:cap], "text", {}


@app.post("/api/analyze")
async def analyze(
    request: Request,
    job_title: str = Form(""),
    job_description: str = Form(""),
    ats_cv_text: str = Form(""),
    exec_cv_text: str = Form(""),
    ats_cv_file: UploadFile | None = None,
    exec_cv_file: UploadFile | None = None,
    claims: str = Form(""),
    prior: str = Form(""),
):
    ip = _client(request)
    if _rate_limited(ip):
        raise HTTPException(429, f"That's {settings.rate_limit_per_hour} analyses this hour from your connection — the limit. Try again later.")

    ats_cv, ats_source, ats_signals = await _read_input("ATS CV", ats_cv_file, ats_cv_text, settings.max_chars_cv)
    exec_cv, exec_source, exec_signals = await _read_input("Executive CV", exec_cv_file, exec_cv_text, settings.max_chars_cv)
    jd_text = (job_description or "").strip()[: settings.max_chars_jd]
    job_title = (job_title or "").strip()[:200]

    if not ats_cv and not exec_cv:
        raise HTTPException(400, "Add a CV — upload a file or paste the text.")
    if len(ats_cv) + len(exec_cv) < 200:
        raise HTTPException(400, "That CV is too short to assess. Paste the full text.")
    if not jd_text and not job_title:
        raise HTTPException(400, "Tell us the job — a title at minimum, ideally the full description.")

    # Re-scan inputs. The previous result is sent back by the browser rather
    # than held here: the service keeps nothing between requests, so the client
    # is the only place it can live, and that keeps the storage promise intact.
    claim_list = [c.strip() for c in (claims or "").split("\n") if c.strip()][:8]
    claim_list = [c[:600] for c in claim_list]
    prior_obj = None
    if prior.strip():
        try:
            parsed = json.loads(prior)
            if isinstance(parsed, dict):
                prior_obj = parsed
        except (json.JSONDecodeError, ValueError):
            prior_obj = None   # a malformed benchmark is ignored, not fatal

    run_id = uuid.uuid4().hex
    _runs[run_id] = {
        "queue": asyncio.Queue(),
        "created": time.time(),
        "args": (ats_cv, exec_cv, job_title, jd_text,
                 ats_source if ats_cv.strip() else exec_source,
                 ats_signals if ats_cv.strip() else exec_signals,
                 claim_list, prior_obj),
        "started": False,
    }
    return {"run_id": run_id}


async def _drive(run_id: str):
    rec = _runs.get(run_id)
    if not rec:
        return
    q: asyncio.Queue = rec["queue"]

    async def progress(msg: str, data: dict | None = None):
        await q.put({"type": "progress", "message": msg, "data": data or {}})

    try:
        async with _run_slots:
            await progress("Warming up the local models")
            a = rec["args"]
            result = await run_council(
                a[0], a[1], a[2], a[3], source=a[4], signals=a[5],
                progress=progress, claim_texts=a[6], prior=a[7],
            )
        await q.put({"type": "result", "result": result.model_dump()})
    except Exception as e:
        await q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        await q.put({"type": "done"})
        # The CV text goes here and nowhere else. Drop it as soon as the run ends.
        rec["args"] = None


@app.get("/api/stream/{run_id}")
async def stream(run_id: str):
    rec = _runs.get(run_id)
    if not rec:
        raise HTTPException(404, "That analysis has expired — start a new one.")
    if rec["started"]:
        raise HTTPException(409, "That analysis is already streaming.")
    rec["started"] = True
    task = asyncio.create_task(_drive(run_id))

    async def gen():
        q: asyncio.Queue = rec["queue"]
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"      # proxies drop idle connections
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["type"] == "done":
                    break
        finally:
            task.cancel()
            _runs.pop(run_id, None)              # nothing survives the request

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.on_event("startup")
async def _prewarm():
    """Pin the council's models in VRAM before the first visitor arrives.

    A model that has to load inside a run adds its load time to that person's
    wait, and the first run after an idle period was consistently the slowest.
    This pays that cost once, at boot, in the background — a failure here is
    irrelevant, because the run path loads models on demand anyway.
    """
    if not settings.prewarm:
        return

    async def warm():
        from council.personas import COUNCIL
        from council.providers import Ollama

        llm = Ollama()
        wanted: list[str] = []
        for m in COUNCIL:
            model = await llm.resolve(m.preferred_models)
            if model and model not in wanted:
                wanted.append(model)
        for model in wanted:
            try:
                # One token is enough to force the weights resident.
                await llm.generate("", "hi", model, max_tokens=1, json_mode=False, timeout=180)
            except Exception:
                pass

    asyncio.create_task(warm())


@app.on_event("startup")
async def _reaper():
    async def sweep():
        while True:
            await asyncio.sleep(120)
            now = time.time()
            for rid, rec in list(_runs.items()):
                if rec["created"] < now - 900:
                    _runs.pop(rid, None)         # abandoned runs must not hold CV text
            # The rate-limit table would otherwise grow one entry per visiting
            # IP for the life of the process.
            for ip, q in list(_hits.items()):
                while q and now - q[0] > 3600:
                    q.popleft()
                if not q:
                    _hits.pop(ip, None)
    asyncio.create_task(sweep())


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
