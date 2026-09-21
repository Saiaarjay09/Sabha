"""Runtime settings. Everything is overridable by environment variable so the
launchd plist can configure a deployment without editing code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"COUNCIL_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))

    # How many council members may be in flight at once. Each in-flight member
    # pins its own model in memory, so this trades latency against VRAM — and
    # the ceiling is real: at 4, four models (~31 GiB) exceeded the 37.4 GiB
    # this machine reports once contexts are added, Ollama started evicting
    # mid-request, and two members died with ReadTimeout. 3 keeps the working
    # set at ~24 GiB and every member returns.
    concurrency: int = field(default_factory=lambda: _env_int("CONCURRENCY", 3))

    # A single member that hangs must not hang the whole run.
    request_timeout_s: float = field(default_factory=lambda: _env_float("TIMEOUT", 300.0))

    # Output budgets. Generation is roughly half of a run's wall time, and a
    # member that writes three crisp strengths is not worse than one that
    # writes four rambling ones — so these are cost levers, not quality ones.
    member_max_tokens: int = field(default_factory=lambda: _env_int("MEMBER_MAX_TOKENS", 800))
    synth_max_tokens: int = field(default_factory=lambda: _env_int("SYNTH_MAX_TOKENS", 1400))

    # Measured member prompts run ~3.5K tokens. 8192 allocated twice the KV
    # cache that was ever used; 6144 keeps headroom for a long CV without
    # paying for space nothing reaches.
    num_ctx: int = field(default_factory=lambda: _env_int("NUM_CTX", 6144))

    # Load the council's models at startup so the first visitor of the day
    # doesn't pay the load cost inside their own run.
    prewarm: bool = field(default_factory=lambda: _env("PREWARM", "1") not in ("0", "false", "no"))

    # Council members generate at most ~800 tokens, so anything past this is a
    # model that has stopped responding, not one that is thinking. Failing fast
    # matters because the run then has time to retry the member on a model that
    # is known to be answering — a long timeout just burns the budget twice.
    member_timeout_s: float = field(default_factory=lambda: _env_float("MEMBER_TIMEOUT", 150.0))

    # Public deployment guards. The site is on a Tailscale Funnel, so it is
    # reachable by anyone with the link.
    max_chars_cv: int = field(default_factory=lambda: _env_int("MAX_CHARS_CV", 60_000))
    max_chars_jd: int = field(default_factory=lambda: _env_int("MAX_CHARS_JD", 30_000))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
    rate_limit_per_hour: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_HOUR", 12))

    # Mounted under a path on the shared Funnel hostname.
    root_path: str = field(default_factory=lambda: _env("ROOT_PATH", ""))
    port: int = field(default_factory=lambda: _env_int("PORT", 8700))


settings = Settings()
