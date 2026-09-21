"""A record of how long each run takes, and nothing else.

This is the one thing in the system that is written to disk, and it is
deliberately narrow. The point of the file is to find out where the five-to-ten
minutes of a run actually goes, so the answer has to survive restarts — but a
performance log is not an excuse to start keeping what the tool promises not to
keep.

WHAT IS RECORDED
  wall-clock durations for the run and for each stage, each council member's
  latency and which model served it, the code version that produced the row,
  and the small structural counts needed to interpret those numbers (how many
  requirements, how many evidence units, how many members answered). Integers,
  model names and a version string.

WHAT IS NEVER RECORDED
  no CV text, no job description text, no job title, no filenames, no scores,
  no verdicts, no IP address, no run id. Nothing here identifies a person or
  reveals what they submitted. A leaked copy of this file would tell you how
  slow the machine is and nothing whatsoever about who used it.

The file lives outside the repository by default and appends one JSON object
per line, so it is trivially greppable and can be deleted at any time without
affecting the service.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path.home() / "Library" / "Logs" / "Sabha" / "timings.jsonl"


def _path() -> Path:
    return Path(os.environ.get("COUNCIL_TIMINGS_PATH", str(DEFAULT_PATH)))


def enabled() -> bool:
    return os.environ.get("COUNCIL_TIMINGS", "1") not in ("0", "false", "no")


@dataclass
class RunTimer:
    """Collects stage durations for one run."""

    t0: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    members: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            # += so a stage entered twice (e.g. a retry pass) accumulates
            # rather than silently reporting only the last attempt.
            self.stages[name] = round(self.stages.get(name, 0.0) + time.perf_counter() - start, 2)

    def member(self, name: str, model: str, latency_s: float, ok: bool) -> None:
        self.members.append({"member": name, "model": model, "latency_s": round(latency_s, 1), "ok": ok})

    def count(self, **kw: int) -> None:
        self.counts.update({k: int(v) for k, v in kw.items()})

    @property
    def total_s(self) -> float:
        return round(time.perf_counter() - self.t0, 2)

    def record(self) -> dict:
        """Build the row and append it. Never raises into the request path."""
        from council import __version__

        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Rows accumulate across code changes, and a median that silently
            # mixes "before" and "after" is worse than no median at all.
            "version": __version__,
            "total_s": self.total_s,
            "stages": self.stages,
            "members": self.members,
            "counts": self.counts,
        }
        if not enabled():
            return row
        try:
            p = _path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception:
            # A performance log must never be able to fail a user's analysis.
            pass
        return row


def load(path: Path | None = None) -> list[dict]:
    p = path or _path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
