"""Summarise the timing log: where does a run actually spend its time?

    python3 analyse_timings.py [path-to-timings.jsonl]

Reads the append-only record written by council/timings.py and prints a
breakdown by stage, by model and by council member, so that optimisation
targets the stage that actually dominates rather than the one that feels slow.
"""

from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from council import timings

STAGE_ORDER = ["job_analysis", "cv_indexing", "requirement_matching", "council", "rescue", "synthesis"]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = min(int(q * (len(xs) - 1)), len(xs) - 1)
    return xs[i]


def bar(frac: float, width: int = 28) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "·" * (width - n)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = timings.load(path)
    if not rows:
        p = path or Path(timings.DEFAULT_PATH)
        print(f"No timing records yet at {p}.\nRun an analysis and try again.")
        return 1

    # Split by code version, since comparing a median across an optimisation
    # is the whole point of keeping this log.
    versions = sorted({r.get("version", "pre-1.1.0") for r in rows})
    if len(versions) > 1:
        print(f"\n{len(rows)} run(s) across {len(versions)} code version(s):")
        for v in versions:
            xs = [r["total_s"] for r in rows if r.get("version", "pre-1.1.0") == v]
            print(f"  {v:12s} n={len(xs):<3d} median {st.median(xs):7.1f}s  fastest {min(xs):7.1f}s")
        latest = versions[-1]
        keep = [r for r in rows if r.get("version", "pre-1.1.0") == latest]
        if len(keep) >= 1:
            print(f"\n  ↓ breakdown below is for {latest} only ({len(keep)} run(s))")
            rows = keep

    totals = [r["total_s"] for r in rows]
    print(f"\n{len(rows)} run(s) recorded · {rows[0]['ts']} → {rows[-1]['ts']}\n")
    print("TOTAL RUNTIME")
    print(f"  median {st.median(totals):7.1f}s   ({st.median(totals)/60:.1f} min)")
    print(f"  fastest{min(totals):7.1f}s   slowest {max(totals):.1f}s")
    if len(totals) > 2:
        print(f"  p90    {pct(totals, 0.9):7.1f}s")

    # ---- stages -----------------------------------------------------------
    by_stage: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r["stages"].items():
            by_stage[k].append(v)

    med_total = st.median(totals)
    print("\nWHERE THE TIME GOES  (median per stage, share of median run)")
    known = [s for s in STAGE_ORDER if s in by_stage] + [s for s in by_stage if s not in STAGE_ORDER]
    for s in known:
        med = st.median(by_stage[s])
        share = med / med_total if med_total else 0
        print(f"  {s:22s} {med:7.1f}s  {share*100:5.1f}%  {bar(share)}")
    accounted = sum(st.median(by_stage[s]) for s in known)
    print(f"  {'(unaccounted)':22s} {med_total - accounted:7.1f}s")

    # ---- models -----------------------------------------------------------
    by_model: dict[str, list[float]] = defaultdict(list)
    fails: dict[str, int] = defaultdict(int)
    for r in rows:
        for m in r.get("members", []):
            by_model[m["model"]].append(m["latency_s"])
            if not m.get("ok", True):
                fails[m["model"]] += 1

    if by_model:
        print("\nCOUNCIL MEMBERS BY MODEL  (median latency per member call)")
        for model, xs in sorted(by_model.items(), key=lambda kv: -st.median(kv[1])):
            f = f"  {fails[model]} failed" if fails[model] else ""
            print(f"  {model:20s} {st.median(xs):6.1f}s  n={len(xs):<3d} slowest {max(xs):5.1f}s{f}")

    # ---- members ----------------------------------------------------------
    by_member: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for m in r.get("members", []):
            by_member[m["member"]].append(m["latency_s"])
    if by_member:
        print("\nSLOWEST MEMBERS  (median)")
        for name, xs in sorted(by_member.items(), key=lambda kv: -st.median(kv[1]))[:7]:
            print(f"  {name:18s} {st.median(xs):6.1f}s")

    # ---- shape ------------------------------------------------------------
    reqs = [r["counts"].get("requirements", 0) for r in rows if r.get("counts")]
    evs = [r["counts"].get("evidence", 0) for r in rows if r.get("counts")]
    ok = [r["counts"].get("members_ok", 0) for r in rows if r.get("counts")]
    tot = [r["counts"].get("members_total", 0) for r in rows if r.get("counts")]
    if reqs:
        print("\nRUN SHAPE")
        print(f"  requirements  median {st.median(reqs):.0f}")
        print(f"  evidence units median {st.median(evs):.0f}")
        if tot:
            print(f"  members answering {st.median(ok):.0f}/{st.median(tot):.0f} median")
    incomplete = sum(1 for r in rows
                     if r["counts"].get("members_ok", 0) < r["counts"].get("members_total", 0))
    if incomplete:
        print(f"  ⚠ {incomplete}/{len(rows)} run(s) lost at least one member — see the rescue stage above")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
