"""Tests for the parts that must be reproducible.

The model-driven stages can only be judged by reading their output, but
everything that turns their output into a number is ordinary code and is
testable — which matters, because those are the parts that decide what the
candidate is actually told.

Run with: python3 test_council.py
"""

from __future__ import annotations

import sys

from council.aggregate import (
    _fences, confidence_label, consensus, corrected_scores, match_percentage,
    member_overall, overall_score, pool_dimensions, verdict_label,
)
from council.atsaudit import audit
from council.evidence import index_cv, render, valid_ids
from council.jd import _normalise_weights
from council.providers import parse_json
from council.schema import DIMENSIONS, MemberScore, Requirement, RequirementVerdict

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


GOOD_CV = """Jane Doe
jane.doe@example.com | +44 7700 900123 | linkedin.com/in/janedoe

PROFESSIONAL EXPERIENCE

Senior Data Engineer, Monzo — Mar 2021 – Jun 2024
- Rebuilt the batch pipeline, cutting nightly runtime 62% and saving $240k/year
- Led a team of 5 engineers migrating 400+ jobs to Airflow
- Reduced data incidents from 14/month to 2/month over 8 months
- Owned the cost model for a 900-node Spark cluster
- Shipped a lineage tool adopted by 30 analysts
- Cut warehouse spend by 31% across two quarters

EDUCATION
BSc Computer Science, Bristol — Sep 2014 – Jun 2017

SKILLS
Python, Spark, Airflow, dbt, SQL, Kafka
"""

BAD_CV = """Curriculum Vitae of John
Responsible for various things at a company
Worked on projects
Helped the team
"""


def test_json_salvage():
    print("\nJSON salvage from imperfect model output")
    check("plain object", parse_json('{"a": 1}') == {"a": 1})
    check("prose preamble", parse_json('Here is my analysis: {"a": 1} thanks') == {"a": 1})
    check("fenced", parse_json('```json\n{"a": [1,2]}\n```') == {"a": [1, 2]})
    check("brace inside string", parse_json('{"a": "has } brace"}') == {"a": "has } brace"})
    check("escaped quote", parse_json(r'{"a": "say \"hi\""}') == {"a": 'say "hi"'})
    check("array", parse_json('[{"a":1}]') == [{"a": 1}])
    check("unsalvageable returns None", parse_json("no json at all") is None)
    check("empty returns None", parse_json("") is None)


def test_ats_discriminates():
    print("\nATS audit")
    good, bad = audit(GOOD_CV, "pdf"), audit(BAD_CV, "text")
    check("good CV scores well", good.score >= 80, f"got {good.score}")
    check("bad CV scores poorly", bad.score <= 40, f"got {bad.score}")
    check("good beats bad by a wide margin", good.score - bad.score > 35)
    check("deterministic", audit(GOOD_CV, "pdf").score == good.score)
    ids = {c["id"] for c in good.checks}
    check("all checks present", {"email", "sections", "dates", "metrics", "layout"} <= ids)
    check("email found in good CV", next(c["status"] for c in good.checks if c["id"] == "email") == "pass")
    check("email missing in bad CV", next(c["status"] for c in bad.checks if c["id"] == "email") == "fail")
    check("weak verbs caught without bullets",
          next(c["status"] for c in bad.checks if c["id"] == "verbs") != "pass")
    check("failed checks carry a fix", all(c["fix"] for c in bad.checks if c["status"] == "fail"))


def test_evidence_is_verbatim():
    print("\nEvidence indexing")
    units = index_cv(GOOD_CV)
    check("units produced", len(units) >= 8, f"got {len(units)}")
    check("ids are unique", len({u.id for u in units}) == len(units))
    check("ids are sequential", [u.id for u in units] == [f"E{i:02d}" for i in range(1, len(units) + 1)])
    # The integrity claim: every unit is text the candidate actually wrote.
    check("every unit is verbatim from the CV", all(u.text in GOOD_CV for u in units))
    check("bullets inherit employer context",
          any(u.employer and u.kind == "achievement" for u in units))
    check("quantified bullets flagged", sum(1 for u in units if u.quantified) >= 4)
    check("education classified", any(u.kind == "education" for u in units))
    check("render includes ids", "[E01]" in render(units))
    check("valid_ids matches", valid_ids(units) == {u.id for u in units})
    check("empty CV is survivable", index_cv("") == [])
    check("cap respected", len(index_cv(GOOD_CV * 20, max_units=12)) == 12)


def _panel(values: dict[str, float]) -> list[MemberScore]:
    return [MemberScore(member=n, model="m", scores={d: v for d in DIMENSIONS}) for n, v in values.items()]


def test_bias_correction():
    print("\nBias correction")
    # The Skeptic (-8) and the Advocate (+7) disagree by 15 raw points. If the
    # correction works, that disagreement should vanish entirely.
    sk = MemberScore(member="skeptic", model="m", scores={d: 42.0 for d in DIMENSIONS})
    ad = MemberScore(member="advocate", model="m", scores={d: 57.0 for d in DIMENSIONS})
    check("skeptic corrected up", corrected_scores(sk)["depth"] == 50.0)
    check("advocate corrected down", corrected_scores(ad)["depth"] == 50.0)
    w = {d: 1 / 6 for d in DIMENSIONS}
    check("raw overalls differ", member_overall(sk, w, corrected=False) != member_overall(ad, w, corrected=False))
    check("corrected overalls agree", member_overall(sk, w) == member_overall(ad, w))
    check("agreement reads as consensus", consensus([sk, ad], w) > 95)
    spread = _panel({"skeptic": 30.0, "advocate": 95.0})
    check("real disagreement lowers consensus", consensus(spread, w) < 60)
    check("correction is clamped to 0-100",
          corrected_scores(MemberScore(member="advocate", model="m", scores={d: 3.0 for d in DIMENSIONS}))["depth"] == 0.0)


def test_outlier_damping():
    print("\nOutlier damping")
    lo, hi = _fences([80, 85, 90, 88, 20, 82, 86])
    check("fence excludes the outlier", lo > 20 and hi > 90, f"got {lo},{hi}")
    check("too few values means no fencing", _fences([10, 90]) == (0.0, 100.0))
    check("identical values means no fencing", _fences([70, 70, 70, 70]) == (0.0, 100.0))
    panel = _panel({"screener": 80, "hiring_manager": 85, "technical": 90,
                    "skeptic": 88, "advocate": 20, "executive": 82, "ats_machine": 86})
    pooled, clamped = pool_dimensions(panel)
    check("outlier was clamped", clamped > 0)
    check("pooled stays near the panel", pooled["risk"] > 70, f"got {pooled['risk']}")
    clean = _panel({"screener": 80, "hiring_manager": 82, "technical": 81,
                    "skeptic": 79, "advocate": 83, "executive": 80, "ats_machine": 81})
    _, clamped2 = pool_dimensions(clean)
    check("agreeing panel is untouched", clamped2 == 0)


def test_match_gate():
    print("\nMatch percentage and the must-have gate")
    reqs = [Requirement(id="R01", text="K8s", criticality="must"),
            Requirement(id="R02", text="Go", criticality="must"),
            Requirement(id="R03", text="Nice", criticality="nice")]

    def v(cov, conf=0.9):
        return [RequirementVerdict(requirement_id=r.id, coverage=c, confidence=conf)
                for r, c in zip(reqs, cov)]

    full, gaps, _ = match_percentage(reqs, v(["direct", "direct", "direct"]))
    check("everything evidenced scores 100", full == 100.0, f"got {full}")
    check("no blocking gaps", gaps == [])

    one, gaps1, raw1 = match_percentage(reqs, v(["absent", "direct", "direct"]))
    check("one missing must is capped at 65", one <= 65.0, f"got {one}")
    check("the gap is named", gaps1 == ["K8s"])
    check("raw coverage is preserved separately", raw1 >= one)

    two, gaps2, _ = match_percentage(reqs, v(["absent", "absent", "direct"]))
    check("two missing musts cap harder", two <= 45.0, f"got {two}")
    check("both gaps named", len(gaps2) == 2)

    trans, _, _ = match_percentage(reqs, v(["transferable", "direct", "direct"]))
    check("transferable earns partial credit, not zero", 70 < trans < 100, f"got {trans}")
    check("transferable does not block", match_percentage(reqs, v(["transferable", "direct", "direct"]))[1] == [])

    sure, _, _ = match_percentage(reqs, v(["direct", "direct", "direct"], conf=0.9))
    unsure, _, _ = match_percentage(reqs, v(["direct", "direct", "direct"], conf=0.2))
    check("low confidence discounts credit", unsure < sure, f"{unsure} vs {sure}")
    check("empty requirements are survivable", match_percentage([], [])[0] == 0.0)


def test_zero_weight_guard():
    """A role that arrives without dimension weights must not score 0.

    Regression: on a re-scan the rubric is reconstructed from the previous
    result, and if the weights were not carried across, the weighted sum
    divided a zero numerator by a fallback denominator and returned a
    confident 0.0 for a candidate the council had scored in the eighties.
    """
    print("\nZero-weight guard")
    pooled = {d: 84.0 for d in DIMENSIONS}
    check("empty weights fall back to the unweighted mean", overall_score(pooled, {}) == 84.0,
          f"got {overall_score(pooled, {})}")
    check("all-zero weights do the same", overall_score(pooled, {d: 0.0 for d in DIMENSIONS}) == 84.0)
    check("real weights still apply", overall_score({**pooled, "depth": 20.0},
          {"depth": 1.0, **{d: 0.0 for d in DIMENSIONS if d != "depth"}}) == 20.0)
    m = MemberScore(member="skeptic", model="m", scores={d: 60.0 for d in DIMENSIONS})
    check("member overall survives empty weights", member_overall(m, {}) > 0)


def test_ats_compliance():
    print("\nATS compliance verdict")
    clean = audit(GOOD_CV, "pdf", {"pages": 2, "images": 0})
    check("a clean CV is compliant", clean.compliant is True, f"blockers {clean.blockers}")
    check("and has no blockers", clean.blockers == [])
    tabled = audit(GOOD_CV, "docx", {"tables": 2, "textboxes": 1, "images": 0})
    check("tables break compliance", tabled.compliant is False)
    check("text boxes are named as blockers", any("text box" in b.lower() for b in tabled.blockers))
    check("compliance is independent of score", tabled.score > 50 and not tabled.compliant,
          f"score {tabled.score}")
    check("structural checks only apply to real files", "tables" not in {c["id"] for c in audit(GOOD_CV, "text").checks})
    check("every check declares criticality", all("critical" in c for c in clean.checks))


def test_gate_has_no_plateau():
    """A gated match must still discriminate.

    Regression: the gate used a hard min(), so every run with one unevidenced
    must-have reported exactly 65.0 regardless of how much else was covered.
    That made the headline uninformative, and because a single must/strong
    reclassification between runs flips the gate, the same CV could read 78%
    then 65% with nothing changed.
    """
    print("\nGate has no plateau")

    def m(cov, crit):
        reqs = [Requirement(id=f"R{i:02d}", text=f"r{i}", criticality=c)
                for i, c in enumerate(crit, 1)]
        vs = [RequirementVerdict(requirement_id=r.id, coverage=c, confidence=0.9)
              for r, c in zip(reqs, cov)]
        return match_percentage(reqs, vs)

    crit = ["must"] + ["strong"] * 5
    strong, _, raw_s = m(["absent"] + ["direct"] * 5, crit)
    weak, _, raw_w = m(["absent"] + ["weak"] * 5, crit)
    check("a gated run still beats a worse gated run", strong > weak, f"{strong} vs {weak}")
    check("the gate still bites hard", strong < raw_s, f"{strong} vs raw {raw_s}")
    check("no flat 65.0 plateau", strong != 65.0 or weak != 65.0)
    mid, _, _ = m(["absent"] + ["direct"] * 3 + ["weak"] * 2, crit)
    check("and it is monotonic in coverage", weak <= mid <= strong, f"{weak} / {mid} / {strong}")
    check("an ungated run is never penalised", m(["direct"] * 6, crit)[0] == 100.0)


def test_labels():
    print("\nVerdict and confidence labels")
    check("strong", verdict_label(85, 85, 0) == "strong fit")
    check("interview", verdict_label(70, 70, 0) == "worth interviewing")
    check("borderline", verdict_label(55, 50, 0) == "borderline")
    check("weak", verdict_label(30, 20, 0) == "not a fit yet")
    check("two missing musts override a high score", verdict_label(95, 95, 2) == "not a fit yet")
    vs = [RequirementVerdict(requirement_id="R01", coverage="direct", confidence=0.95)]
    check("high confidence", confidence_label(90, vs, 7, 7) == "high")
    lo = [RequirementVerdict(requirement_id="R01", coverage="direct", confidence=0.2)]
    check("low confidence", confidence_label(30, lo, 3, 7) == "low")


def test_weights():
    print("\nRole-adaptive dimension weights")
    w = _normalise_weights({"relevance": 5, "depth": 3, "impact": 2})
    check("sums to 1", abs(sum(w.values()) - 1.0) < 0.01, f"got {sum(w.values())}")
    check("all dimensions present", set(w) == set(DIMENSIONS))
    check("no dimension is zeroed out", all(v >= 0.015 for v in w.values()))
    check("ordering preserved", w["relevance"] > w["depth"] > w["impact"])
    check("garbage falls back", abs(sum(_normalise_weights(None).values()) - 1.0) < 0.01)
    check("all-zero falls back", abs(sum(_normalise_weights({d: 0 for d in DIMENSIONS}).values()) - 1.0) < 0.01)
    pooled = {d: 80.0 for d in DIMENSIONS}
    check("uniform scores give that score", abs(overall_score(pooled, w) - 80.0) < 0.5)


def main():
    for t in (test_json_salvage, test_ats_discriminates, test_evidence_is_verbatim,
              test_bias_correction, test_outlier_damping, test_match_gate,
              test_zero_weight_guard, test_ats_compliance, test_gate_has_no_plateau,
              test_labels, test_weights):
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'All checks passed.'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
