"""Run a full assessment.

Order matters and is the argument for the whole design:

  1. Read the JOB first and decompose it into requirements, each with the
     evidence that would prove it and the proxies that count. No CV involved.
  2. Index the CV into citable evidence, mechanically.
  3. Judge each requirement against that evidence with two models.
  4. Only then convene the council, which sees the role, the evidence and the
     requirement verdicts, and argues about the candidate as a whole.
  5. Aggregate with bias correction, then synthesise concrete fixes.

Nothing is written to disk at any point. The CV exists in memory for the life
of the request and is discarded when it returns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from council import aggregate, atsaudit, evidence, jd, match
from council.config import settings
from council.personas import BY_NAME, COUNCIL, Member, system_prompt
from council.providers import Ollama
from council.schema import (
    COVERAGE_CREDIT,
    DIMENSIONS,
    ATSAudit,
    CouncilResult,
    Improvement,
    MemberScore,
    RequirementVerdict,
    RoleProfile,
)

Progress = Callable[[str, dict | None], Awaitable[None]]

_MEMBER_TEMPLATE = """
ROLE BEING ASSESSED: {title} ({seniority}{family})
WHAT THIS ROLE ACTUALLY NEEDS:
{summary}

REQUIREMENT ANALYSIS ALREADY PERFORMED (independent of you).
This is what the JOB demands and whether the CV evidences it. It is NOT a list
of the candidate's skills — read each VERDICT carefully before you characterise
what this person has done:
{req_summary}

CV EVIDENCE (cite these ids):
{evidence}

{cv_section}

Score the candidate FOR THIS ROLE on each dimension, 0-100:
  relevance     — how much this experience bears on this specific job
  depth         — how credible and deep the demonstrated expertise is
  impact        — outcomes, scope, and measurable results
  trajectory    — direction and rate of growth, and where it points next
  communication — how well the CV itself makes the candidate's case
  risk          — 100 means no red flags at all, 0 means serious concerns

Return exactly:

{{
  "scores": {{"relevance": 0, "depth": 0, "impact": 0, "trajectory": 0, "communication": 0, "risk": 0}},
  "headline": "one sentence, your verdict in your own voice",
  "strengths": ["2-4 specific strengths, each citing evidence ids"],
  "concerns": ["2-4 specific concerns, each citing evidence ids where relevant"],
  "recommendation": "strong_yes | yes | borderline | no"
}}
""".strip()

_SYNTH_SYSTEM = """
You are a senior CV coach. You are given a council's assessment of one CV
against one specific job, including which requirements the candidate failed to
evidence and which mechanical ATS checks failed.

Write fixes the candidate can actually apply this afternoon. Every item must
name a specific problem and a specific action. "Add more detail" is useless;
"Your Monzo bullet says you led a migration but never says how many services or
how long it took — give both" is useful.

Never suggest claiming experience they lack, and never phrase an action in a
way that assumes experience the evidence does not show — "add a bullet about
your Terraform project" is exactly that mistake if there is no Terraform
project. For a genuine gap, do one of two things and say which:
  (a) name the adjacent experience they DO have in the evidence and show how to
      surface it honestly, making the transfer explicit rather than implied; or
  (b) state plainly that this is a real gap the CV cannot fix, and say what
      would close it outside the CV.
Distinguish the two. A candidate who cannot tell "you already have this, bury
it less" from "you do not have this" cannot act on your advice.

Output ONLY the JSON object. No preamble, no markdown fence.
""".strip()

_SYNTH_TEMPLATE = """
ROLE: {title} ({seniority})

REQUIREMENTS THE CANDIDATE DID NOT FULLY EVIDENCE:
{gaps}

WHAT THE COUNCIL WAS CONCERNED ABOUT:
{concerns}

MECHANICAL ATS CHECKS THAT FAILED:
{ats}

THE CANDIDATE'S CV EVIDENCE:
{evidence}

Return exactly:

{{
  "summary": "3-4 sentences to the candidate, plain and direct: where they stand for this role and what most needs to change. Address them as 'you'.",
  "improvements": [
    {{
      "priority": "critical | high | medium",
      "area": "short label, e.g. 'Missing evidence: scale'",
      "problem": "what is wrong right now, referencing their actual CV",
      "action": "exactly what to do about it",
      "example": "a concrete rewritten line they could adapt, or an empty string"
    }}
  ]
}}

Give between 5 and 8 improvements, most important first.
""".strip()

_SYNTH_MODELS = ("qwen2.5:14b", "phi4:14b", "mistral-nemo:12b", "llama3.1:8b", "llama3.2:3b")


# Spelled out candidate-relative rather than as a bare label. Smaller models
# read "Kubernetes → TRANSFERABLE" as "the candidate knows Kubernetes" and
# then repeat it as fact; saying "has NOT done this" leaves no room for that.
_COVERAGE_GLOSS = {
    "direct": "EVIDENCED — the candidate has demonstrably done this",
    "transferable": "NOT DIRECTLY EVIDENCED — the candidate has NOT done this specific thing, but adjacent experience may carry over",
    "partial": "PARTIALLY EVIDENCED — some of this, clearly not all of it",
    "weak": "BARELY EVIDENCED — only a passing mention or a bare skills-list entry",
    "absent": "NOT EVIDENCED AT ALL — nothing in the CV supports this",
}


def _req_summary(role: RoleProfile, verdicts: list[RequirementVerdict]) -> str:
    by_id = {v.requirement_id: v for v in verdicts}
    out = []
    for r in role.requirements:
        v = by_id.get(r.id)
        cov = v.coverage if v else "absent"
        cites = f" Supporting evidence: {', '.join(v.evidence_ids)}." if v and v.evidence_ids else ""
        line = f"- THE JOB ASKS FOR ({r.criticality.upper()}): {r.text}\n  VERDICT: {_COVERAGE_GLOSS[cov]}.{cites}"
        if v and v.reasoning:
            # Truncated: every member pays prompt-processing time for this
            # block, and the first clause carries the signal.
            basis = v.reasoning if len(v.reasoning) <= 170 else v.reasoning[:167].rsplit(" ", 1)[0] + "…"
            line += f"\n  Basis: {basis}"
        out.append(line)
    return "\n".join(out)


async def _run_member(
    llm: Ollama, m: Member, model: str, role: RoleProfile, ev_block: str, req_summary: str,
    ats_cv: str, exec_cv: str, sem: asyncio.Semaphore, progress: Progress | None,
) -> MemberScore:
    # Either document may be missing, and a member must never be handed a blank
    # one: the ATS readers fall back to the narrative CV and vice versa.
    primary = ats_cv if ats_cv.strip() else exec_cv
    narrative = exec_cv if exec_cv.strip() else ats_cv

    if m.reads == "ats" or not exec_cv.strip():
        cv_section = f"THE CV AS SUBMITTED (ATS version):\n\"\"\"\n{primary}\n\"\"\""
    elif m.reads == "exec":
        cv_section = f"THE CANDIDATE'S NARRATIVE / EXECUTIVE CV:\n\"\"\"\n{narrative}\n\"\"\""
    elif not ats_cv.strip():
        cv_section = f"THE CANDIDATE'S NARRATIVE / EXECUTIVE CV:\n\"\"\"\n{narrative}\n\"\"\""
    else:
        cv_section = (
            f"THE CV AS SUBMITTED (ATS version):\n\"\"\"\n{ats_cv}\n\"\"\"\n\n"
            f"THE CANDIDATE'S NARRATIVE / EXECUTIVE CV:\n\"\"\"\n{exec_cv}\n\"\"\""
        )

    async with sem:
        if progress:
            await progress(f"{m.title} is reading ({model})", {"member": m.name, "state": "running"})
        data, resp = await llm.generate_json(
            system_prompt(m),
            _MEMBER_TEMPLATE.format(
                title=role.title, seniority=role.seniority,
                family=f", {role.role_family}" if role.role_family else "",
                summary=role.summary or "(not stated)",
                req_summary=req_summary, evidence=ev_block, cv_section=cv_section,
            ),
            model, temperature=m.temperature, max_tokens=1100,
            timeout=settings.member_timeout_s,
        )

    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        return MemberScore(member=m.name, model=model, latency_s=round(resp.latency_s, 1),
                           error=resp.error or "returned no usable scores")

    scores: dict[str, float] = {}
    for d in DIMENSIONS:
        try:
            scores[d] = max(0.0, min(100.0, float(data["scores"].get(d, 50))))
        except (TypeError, ValueError):
            scores[d] = 50.0

    def _list(key: str) -> list[str]:
        v = data.get(key) or []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip()[:400] for x in v if str(x).strip()][:4]

    rec = str(data.get("recommendation", "borderline")).lower().strip().replace(" ", "_")
    return MemberScore(
        member=m.name, model=model, scores=scores,
        headline=str(data.get("headline", "")).strip()[:300],
        strengths=_list("strengths"), concerns=_list("concerns"),
        recommendation=rec if rec in {"strong_yes", "yes", "borderline", "no"} else "borderline",
        latency_s=round(resp.latency_s, 1),
    )


def _deterministic_improvements(ats: ATSAudit) -> list[Improvement]:
    """Fixes that need no model — the mechanical failures speak for themselves."""
    out = []
    for c in ats.checks:
        if c["status"] == "fail" and c.get("fix"):
            out.append(Improvement(
                priority="critical" if c["weight"] >= 2.5 else "high",
                area=f"ATS: {c['label']}", problem=c["detail"], action=c["fix"],
            ))
    return out


async def _synthesise(
    llm: Ollama, role: RoleProfile, verdicts: list[RequirementVerdict],
    members: list[MemberScore], ats: ATSAudit, ev_block: str,
) -> tuple[str, list[Improvement]]:
    by_id = {r.id: r for r in role.requirements}
    gaps = [
        f"- ({by_id[v.requirement_id].criticality.upper()}) {by_id[v.requirement_id].text} "
        f"→ {v.coverage.upper()}. {v.gap or v.reasoning}"
        for v in verdicts
        if v.requirement_id in by_id and COVERAGE_CREDIT.get(v.coverage, 0) < 0.75
    ][:10]
    concerns = [f"- {c}" for m in members if m.ok for c in m.concerns][:14]
    failed = [f"- {c['label']}: {c['detail']}" for c in ats.checks if c["status"] != "pass"][:10]

    model = await llm.resolve(_SYNTH_MODELS)
    det = _deterministic_improvements(ats)
    if not model:
        return "", det

    data, _ = await llm.generate_json(
        _SYNTH_SYSTEM,
        _SYNTH_TEMPLATE.format(
            title=role.title, seniority=role.seniority,
            gaps="\n".join(gaps) or "(none — the candidate evidenced everything)",
            concerns="\n".join(concerns) or "(none recorded)",
            ats="\n".join(failed) or "(all mechanical checks passed)",
            evidence=ev_block,
        ),
        model, temperature=0.35, max_tokens=2200,
    )
    if not isinstance(data, dict):
        return "", det

    improvements: list[Improvement] = []
    for it in data.get("improvements") or []:
        if not isinstance(it, dict) or not str(it.get("action", "")).strip():
            continue
        pri = str(it.get("priority", "medium")).lower().strip()
        improvements.append(Improvement(
            priority=pri if pri in {"critical", "high", "medium"} else "medium",
            area=str(it.get("area", "")).strip()[:120],
            problem=str(it.get("problem", "")).strip()[:600],
            action=str(it.get("action", "")).strip()[:600],
            example=str(it.get("example", "")).strip()[:600],
        ))

    # Mechanical failures the coach didn't mention are appended, not replaced:
    # they're reproducible facts and shouldn't depend on a model remembering.
    seen = " ".join((i.area + i.action).lower() for i in improvements)
    for d in det:
        key = d.area.split(":", 1)[-1].strip().lower()
        if key and key not in seen:
            improvements.append(d)

    order = {"critical": 0, "high": 1, "medium": 2}
    improvements.sort(key=lambda i: order.get(i.priority, 3))
    return str(data.get("summary", "")).strip()[:1200], improvements[:10]


async def run(
    ats_cv: str, exec_cv: str, job_title: str, jd_text: str, source: str = "text",
    progress: Progress | None = None,
) -> CouncilResult:
    t0 = time.perf_counter()
    llm = Ollama()
    notes: list[str] = []

    async def say(msg: str, data: dict | None = None):
        if progress:
            await progress(msg, data)

    installed = await llm.available_models()
    if not installed:
        raise RuntimeError("No local models are available — the Ollama service isn't reachable.")

    # 1. the job, before the CV
    await say("Reading the job and working out what it actually requires")
    role, note = await jd.analyse_job(llm, job_title, jd_text)
    if note:
        notes.append(note)
    await say(f"Identified {len(role.requirements)} real requirements", {"role": role.model_dump()})

    # 2. the CV, indexed mechanically
    units = evidence.index_cv(ats_cv if ats_cv.strip() else exec_cv)
    ats = atsaudit.audit(ats_cv if ats_cv.strip() else exec_cv, source)
    await say(f"Indexed {len(units)} pieces of evidence from the CV")
    ev_block = evidence.render(units)

    # 3. requirement-by-requirement judgement
    verdicts, mnotes = await match.match_requirements(llm, role, units, lambda m: say(m))
    notes += mnotes
    await say("Requirement matching complete", {"verdicts": [v.model_dump() for v in verdicts]})

    # 4. the council
    req_summary = _req_summary(role, verdicts)
    sem = asyncio.Semaphore(max(1, settings.concurrency))
    await say(f"Convening the council — {len(COUNCIL)} assessors")

    # Group members by the model they resolved to, and run one group at a time.
    #
    # Running all seven members freely meant Ollama had to evict a model that
    # was still generating in order to load the next member's — and the
    # displaced requests sat in the queue until they timed out, losing whole
    # members from the panel. Grouping keeps at most one model loading at any
    # moment: every member in a group runs against weights that are already
    # resident, and the next model is only fetched once the previous group is
    # done with the GPU.
    groups: dict[str, list[Member]] = {}
    unavailable: list[MemberScore] = []
    for m in COUNCIL:
        model = await llm.resolve(m.preferred_models)
        if not model:
            unavailable.append(MemberScore(member=m.name, model="none", error="no local model available"))
            continue
        groups.setdefault(model, []).append(m)

    members: list[MemberScore] = list(unavailable)
    # Busiest model first: it loads once and serves the most members.
    for model, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        members += list(await asyncio.gather(*[
            _run_member(llm, m, model, role, ev_block, req_summary, ats_cv, exec_cv, sem, progress)
            for m in group
        ]))

    # Second pass: rescue members that failed.
    #
    # Under GPU pressure an individual model can stop responding and take its
    # member's whole assessment with it — and a panel silently missing its
    # Skeptic or its Advocate is not the panel this tool claims to run, because
    # bias correction assumes the opposing views are both present. So any
    # member that failed is retried once on a model that has already answered
    # successfully in THIS run, and the substitution is recorded rather than
    # hidden: diversity is the goal, but a present member on a borrowed model
    # beats an absent one.
    proven = [ms.model for ms in members if ms.ok]
    failed = [ms for ms in members if not ms.ok and ms.model != "none"]
    if failed and proven:
        rescue_model = max(set(proven), key=proven.count)
        await say(f"Retrying {len(failed)} member(s) that timed out, on {rescue_model}")
        retried = list(await asyncio.gather(*[
            _run_member(llm, BY_NAME[ms.member], rescue_model, role, ev_block, req_summary,
                        ats_cv, exec_cv, sem, progress)
            for ms in failed if ms.member in BY_NAME
        ]))
        recovered = 0
        by_name = {ms.member: ms for ms in members}
        for r in retried:
            if r.ok:
                by_name[r.member] = r
                recovered += 1
        members = list(by_name.values())
        if recovered:
            notes.append(f"{recovered} member(s) failed on their own model and were re-run on {rescue_model}.")

    # Restore the declared council order so the UI is stable run to run.
    order = {m.name: i for i, m in enumerate(COUNCIL)}
    members.sort(key=lambda ms: order.get(ms.member, 99))
    ok = [m for m in members if m.ok]
    if not ok:
        raise RuntimeError("Every council member failed to return a usable assessment.")
    if len(ok) < len(members):
        notes.append(f"{len(members) - len(ok)} of {len(members)} council members failed and were excluded.")

    # 5. aggregate, then coach
    dim_weights = role.dimension_weights
    pooled, clamped = aggregate.pool_dimensions(members)
    score = aggregate.overall_score(pooled, dim_weights)
    match_pct, blocking, raw_match = aggregate.match_percentage(role.requirements, verdicts)
    cons = aggregate.consensus(members, dim_weights)
    if clamped:
        notes.append(f"{clamped} outlying dimension score(s) were pulled back to the panel's range before pooling.")
    if blocking and raw_match > match_pct:
        notes.append(f"Requirement coverage was {raw_match:.0f}% but is capped at {match_pct:.0f}% because a must-have is unevidenced.")

    await say("Aggregating the panel and writing your improvements")
    summary, improvements = await _synthesise(llm, role, verdicts, members, ats, ev_block)

    return CouncilResult(
        score=score,
        match_pct=match_pct,
        verdict=aggregate.verdict_label(score, match_pct, len(blocking)),
        confidence=aggregate.confidence_label(cons, verdicts, len(ok), len(members)),
        consensus=cons,
        role=role,
        dimensions=pooled,
        members=members,
        requirements=verdicts,
        evidence=units,
        improvements=improvements,
        ats=ats,
        blocking_gaps=blocking,
        summary=summary,
        elapsed_s=round(time.perf_counter() - t0, 1),
        notes=notes,
    )
