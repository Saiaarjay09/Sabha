"""Turn a disagreeing panel into two defensible numbers.

Three ideas do the work here.

1. Bias correction. Every member has a declared lean (`offset`) and it is
   subtracted before anything is pooled. The Skeptic cannot drag the panel down
   by being the Skeptic, and the Advocate cannot lift it by being the Advocate.
   What survives correction is the part of their judgement that is about the
   candidate rather than about the persona.

2. Competence-weighted pooling. Each dimension is pooled using each member's
   own weight on that dimension, times that member's reliability. The ATS
   Parser therefore dominates machine-readability and barely touches impact,
   which is right: a member should carry weight where it was designed to look.

3. A gate, not just an average, on the match percentage. Averaging lets four
   satisfied "nice to have" requirements paper over a missing "must". Real
   hiring does not work that way, so a missing must-have caps the headline
   number and the reason for the cap is reported.
"""

from __future__ import annotations

import math

from council.schema import (
    COVERAGE_CREDIT,
    CRITICALITY_WEIGHT,
    DIMENSIONS,
    MemberScore,
    Requirement,
    RequirementVerdict,
)
from council.personas import BY_NAME


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def corrected_scores(m: MemberScore) -> dict[str, float]:
    """A member's scores with its declared lean removed."""
    persona = BY_NAME.get(m.member)
    off = persona.offset if persona else 0.0
    return {d: _clamp(float(m.scores.get(d, 50.0)) - off) for d in DIMENSIONS}


def member_overall(m: MemberScore, dim_weights: dict[str, float], corrected: bool = True) -> float:
    """One member's single number, on the role's own axes."""
    s = corrected_scores(m) if corrected else {d: _clamp(float(m.scores.get(d, 50.0))) for d in DIMENSIONS}
    tw = sum(dim_weights.get(d, 0.0) for d in DIMENSIONS) or 1.0
    return round(sum(s[d] * dim_weights.get(d, 0.0) for d in DIMENSIONS) / tw, 1)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _fences(vals: list[float]) -> tuple[float, float]:
    """Tukey fences. Values outside them are pulled back to the fence.

    One member occasionally returns a wild score on one axis — a 20 for risk
    against a panel sitting between 70 and 95. That is a model slip, not a
    minority opinion worth preserving, and on a heavily weighted axis a single
    such value would visibly move the headline. Clamping to the fences keeps
    the dissent (the value stays lowest) without letting it dominate.
    """
    if len(vals) < 4:
        return 0.0, 100.0
    sv = sorted(vals)
    q1, q3 = _quantile(sv, 0.25), _quantile(sv, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0, 100.0
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def pool_dimensions(members: list[MemberScore]) -> tuple[dict[str, float], int]:
    """Pool each dimension across members, weighting each member where it looks.

    Returns (pooled scores, number of outlier values that were clamped).
    """
    pooled: dict[str, float] = {}
    usable = [m for m in members if m.ok]
    clamped = 0
    for d in DIMENSIONS:
        raw = [(m, corrected_scores(m)[d]) for m in usable if BY_NAME.get(m.member)]
        lo, hi = _fences([v for _, v in raw])
        num = den = 0.0
        for m, v in raw:
            persona = BY_NAME[m.member]
            w = persona.reliability * persona.weights.get(d, 0.0)
            if w <= 0:
                continue
            adj = min(max(v, lo), hi)
            if adj != v:
                clamped += 1
            num += adj * w
            den += w
        pooled[d] = round(num / den, 1) if den else 50.0
    return pooled, clamped


def consensus(members: list[MemberScore], dim_weights: dict[str, float]) -> float:
    """How much the panel agreed, 0-100, after bias correction.

    Spread that survives bias correction is genuine disagreement about the
    candidate, and it is the honest signal for how much to trust the headline.
    """
    vals = [member_overall(m, dim_weights) for m in members if m.ok]
    if len(vals) < 2:
        return 50.0
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    # ~18 points of spread is total disagreement in practice.
    return round(_clamp(100.0 - (sd / 18.0) * 100.0), 1)


def overall_score(pooled: dict[str, float], dim_weights: dict[str, float]) -> float:
    tw = sum(dim_weights.get(d, 0.0) for d in DIMENSIONS) or 1.0
    return round(_clamp(sum(pooled.get(d, 50.0) * dim_weights.get(d, 0.0) for d in DIMENSIONS) / tw), 1)


def match_percentage(
    requirements: list[Requirement], verdicts: list[RequirementVerdict]
) -> tuple[float, list[str], float]:
    """Evidence-weighted requirement coverage, gated on must-haves.

    Returns (match %, blocking gaps, raw uncapped coverage).
    """
    by_id = {v.requirement_id: v for v in verdicts}
    num = den = 0.0
    missing_musts: list[str] = []

    for r in requirements:
        v = by_id.get(r.id)
        credit = COVERAGE_CREDIT.get(v.coverage, 0.0) if v else 0.0
        w = CRITICALITY_WEIGHT.get(r.criticality, 1.0)
        # A verdict the models were unsure about shouldn't score as confidently
        # as one they agreed on; pull low-confidence credit toward neutral.
        if v and v.confidence < 0.5 and credit > 0:
            credit *= 0.5 + v.confidence
        num += credit * w
        den += w
        if r.criticality == "must" and credit < 0.34:
            missing_musts.append(r.text)

    raw = round(100.0 * num / den, 1) if den else 0.0

    # The gate. One unmet must-have means this is not a strong match however
    # well the rest scored; two or more means it is a weak one.
    if len(missing_musts) == 1:
        capped = min(raw, 65.0)
    elif len(missing_musts) >= 2:
        capped = min(raw, 45.0 - 3.0 * (len(missing_musts) - 2))
    else:
        capped = raw
    return round(_clamp(capped), 1), missing_musts, raw


def verdict_label(score: float, match: float, missing_musts: int) -> str:
    if missing_musts >= 2:
        return "not a fit yet"
    if score >= 78 and match >= 75:
        return "strong fit"
    if score >= 65 and match >= 62:
        return "worth interviewing"
    if score >= 50 and match >= 45:
        return "borderline"
    return "not a fit yet"


def confidence_label(consensus_score: float, verdicts: list[RequirementVerdict], n_ok: int, n_total: int) -> str:
    avg_conf = (sum(v.confidence for v in verdicts) / len(verdicts)) if verdicts else 0.5
    panel = n_ok / n_total if n_total else 0.0
    composite = (consensus_score / 100.0) * 0.45 + avg_conf * 0.35 + panel * 0.20
    if composite >= 0.72:
        return "high"
    if composite >= 0.55:
        return "medium"
    return "low"
