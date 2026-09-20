"""Model transport. Open-weight and local only.

CVs are other people's personal data and this site is on a public URL, so the
one hard rule here is that a CV never leaves the machine: every call goes to a
local Ollama daemon. There is no hosted-API path in this file, deliberately.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx

from council.config import settings


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    latency_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict | list | None:
    """Get a JSON value out of whatever the model actually produced.

    Small open-weight models wrap JSON in prose or fences even when asked not
    to, and they run objects together. Rather than fail the whole council
    because one member added "Here is my analysis:", try progressively looser
    reads and give up only if none of them yield valid JSON.
    """
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Balanced-delimiter scan: take the first complete {...} or [...] value,
    # ignoring braces that appear inside strings.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


class Ollama:
    """Async client over the local Ollama daemon."""

    def __init__(self, host: str | None = None, timeout: float | None = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.timeout = timeout or settings.request_timeout_s
        self._models: list[str] | None = None
        self._lock = asyncio.Lock()

    async def available_models(self) -> list[str]:
        async with self._lock:
            if self._models is not None:
                return self._models
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    r = await c.get(f"{self.host}/api/tags")
                    r.raise_for_status()
                    self._models = [m["name"] for m in r.json().get("models", [])]
            except Exception:
                self._models = []
            return self._models

    async def resolve(self, preferred: tuple[str, ...] | list[str]) -> str | None:
        """First preferred model actually installed, matched loosely.

        `qwen2.5:14b` should match an installed `qwen2.5:14b-instruct-q4_K_M`,
        and a council should still convene (with a recorded substitution) when
        a member's first choice was never pulled.
        """
        have = await self.available_models()
        if not have:
            return None
        for want in preferred:
            for h in have:
                if h == want:
                    return h
            base, _, tag = want.partition(":")
            for h in have:
                if h.startswith(base + ":") and tag and tag in h:
                    return h
            for h in have:
                if h.split(":")[0] == base:
                    return h
        return have[0]

    async def generate(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2200,
        json_mode: bool = True,
        timeout: float | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
            # Keep weights resident between members, or a five-model council
            # pays the load cost five times over.
            "keep_alive": "15m",
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as c:
                r = await c.post(f"{self.host}/api/generate", json=payload)
                r.raise_for_status()
                d = r.json()
            return LLMResponse(text=d.get("response", ""), model=model, latency_s=time.perf_counter() - t0)
        except Exception as e:  # a dead member is survivable; a dead council is not
            return LLMResponse(text="", model=model, latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")

    async def generate_json(self, system: str, user: str, model: str, **kw) -> tuple[dict | list | None, LLMResponse]:
        resp = await self.generate(system, user, model, **kw)
        if not resp.ok:
            return None, resp
        return parse_json(resp.text), resp


ollama = Ollama()
