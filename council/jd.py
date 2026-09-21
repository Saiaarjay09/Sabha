"""Decompose a job into what it actually requires.

This runs before the CV is ever read, and that ordering is the point. The model
commits up front to what each requirement means, what evidence would prove it,
and which adjacent experience it is prepared to accept as a proxy. Matching
later is then a judgement against a rubric the model cannot retrofit to
flatter (or punish) the specific candidate in front of it.

It is also where "what the job says" is separated from "what the job needs".
Job descriptions are padded with boilerplate and with requirements no-one
enforces; a requirement list that takes every line at face value reproduces the
same broken filter this tool exists to replace.
"""

from __future__ import annotations

from council.providers import Ollama
from council.schema import DIMENSIONS, Requirement, RoleProfile

_SYSTEM = """
You are an expert technical recruiter and job analyst. You read a job
description and work out what the role ACTUALLY requires day to day — not
merely what the text lists.

You know that job descriptions are unreliable documents: they pad requirement
lists, inherit boilerplate from older postings, ask for more years than the work
needs, and list tools that are incidental to the real problem. Your job is to
recover the underlying role.

For every requirement you must state:
  - `proof`: what a CV would have to show for this to count as demonstrated.
  - `proxies`: different experience that genuinely carries over. This is the
    most important field you write. A person who has run large Postgres
    migrations can almost certainly run MySQL ones; a person who has managed a
    P&L can budget; a teacher can train. Be specific and be realistic — list
    experience that a good hiring manager would actually accept, not anything
    vaguely adjacent.

Mark criticality honestly:
  - "must": the role genuinely fails without it.
  - "strong": materially affects performance, but a strong candidate could
    learn it quickly.
  - "nice": listed, but truly optional.
Most job descriptions have only two to five real "must" items. If you mark
everything "must", you have not done the analysis.

Output ONLY a JSON object. No preamble, no markdown fence.
""".strip()

_TEMPLATE = """
JOB TITLE THE CANDIDATE IS APPLYING FOR: {job_title}

JOB DESCRIPTION / ROLE DETAILS:
\"\"\"
{jd}
\"\"\"

Return this exact JSON shape:

{{
  "title": "the role's actual title",
  "seniority": "one of: intern, junior, mid, senior, lead, principal, executive",
  "role_family": "short family, e.g. 'backend engineering', 'financial controlling'",
  "summary": "2-3 sentences on what this job really needs someone to be able to do. Describe the work, not the posting.",
  "dimension_weights": {{
    "relevance": 0.0, "depth": 0.0, "impact": 0.0,
    "trajectory": 0.0, "communication": 0.0, "risk": 0.0
  }},
  "unstated_expectations": ["things this role will clearly demand that the description does not say out loud"],
  "requirements": [
    {{
      "id": "R01",
      "text": "the requirement, in plain words",
      "category": "one of: skill, experience, domain, credential, behavioural, logistical",
      "criticality": "one of: must, strong, nice",
      "proof": "what evidence in a CV would demonstrate this",
      "proxies": ["adjacent experience that legitimately carries over"]
    }}
  ]
}}

`dimension_weights` must sum to 1.0 and must reflect THIS role: an executive
hire is weighted toward impact and trajectory, a graduate role toward
trajectory and communication, a specialist role toward depth. "risk" is how
much this particular role punishes a bad hire.

Produce between 8 and 16 requirements, ordered most critical first. If the job
description is thin, infer the standard requirements for this title and
seniority from the title itself, and say so in `summary`.
""".strip()

_MODELS = ("qwen2.5:14b", "phi4:14b", "mistral-nemo:12b", "llama3.1:8b", "llama3.2:3b")

_FALLBACK_WEIGHTS = {
    "relevance": 0.26, "depth": 0.22, "impact": 0.22,
    "trajectory": 0.12, "communication": 0.10, "risk": 0.08,
}


def default_weights() -> dict[str, float]:
    """The generic dimension weighting, for when a role carries none."""
    return dict(_FALLBACK_WEIGHTS)


def _normalise_weights(raw: dict | None) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(_FALLBACK_WEIGHTS)
    out: dict[str, float] = {}
    for d in DIMENSIONS:
        try:
            v = float(raw.get(d, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        out[d] = max(v, 0.0)
    total = sum(out.values())
    if total <= 0:
        return dict(_FALLBACK_WEIGHTS)
    # Floor every dimension so a model that zeroes one out cannot make a whole
    # axis of the assessment silently disappear.
    out = {d: max(v / total, 0.02) for d, v in out.items()}
    total = sum(out.values())
    return {d: round(v / total, 4) for d, v in out.items()}


async def analyse_job(llm: Ollama, job_title: str, jd_text: str) -> tuple[RoleProfile, str | None]:
    """Turn a posting into a structured rubric. Returns (profile, note)."""
    model = await llm.resolve(_MODELS)
    if not model:
        return _heuristic_profile(job_title), "No local model available — used a generic rubric for this title."

    data, resp = await llm.generate_json(
        _SYSTEM,
        _TEMPLATE.format(job_title=job_title or "(not stated)", jd=jd_text.strip() or "(none supplied)"),
        model,
        # Zero, not 0.2. This step decides how many requirements exist and
        # which are "must" — and a single must/strong flip changes whether the
        # must-have gate fires, which moves the headline match by more than
        # ten points. Measured on one job description, repeated runs produced
        # five must-haves one time and six the next, so the same CV scored 78%
        # and then 65%. The rubric has to be the stable part.
        temperature=0.0,
        max_tokens=2600,
    )
    if not isinstance(data, dict) or not data.get("requirements"):
        return _heuristic_profile(job_title), f"Job analysis fell back to a generic rubric ({resp.error or 'unparseable model output'})."

    reqs: list[Requirement] = []
    for i, r in enumerate(data.get("requirements") or [], 1):
        if not isinstance(r, dict) or not str(r.get("text", "")).strip():
            continue
        crit = str(r.get("criticality", "strong")).lower().strip()
        cat = str(r.get("category", "skill")).lower().strip()
        proxies = r.get("proxies") or []
        if isinstance(proxies, str):
            proxies = [proxies]
        reqs.append(Requirement(
            id=f"R{i:02d}",
            text=str(r["text"]).strip()[:300],
            category=cat if cat in {"skill", "experience", "domain", "credential", "behavioural", "logistical"} else "skill",
            criticality=crit if crit in {"must", "strong", "nice"} else "strong",
            proof=str(r.get("proof", "")).strip()[:300],
            proxies=[str(p).strip()[:160] for p in proxies if str(p).strip()][:6],
        ))
    if not reqs:
        return _heuristic_profile(job_title), "Job analysis produced no usable requirements — used a generic rubric."

    sen = str(data.get("seniority", "mid")).lower().strip()
    unstated = data.get("unstated_expectations") or []
    if isinstance(unstated, str):
        unstated = [unstated]

    profile = RoleProfile(
        title=str(data.get("title") or job_title or "the role").strip()[:160],
        seniority=sen if sen in {"intern", "junior", "mid", "senior", "lead", "principal", "executive"} else "mid",
        role_family=str(data.get("role_family", "")).strip()[:120],
        summary=str(data.get("summary", "")).strip()[:900],
        requirements=reqs,
        dimension_weights=_normalise_weights(data.get("dimension_weights")),
        unstated_expectations=[str(u).strip()[:200] for u in unstated if str(u).strip()][:6],
    )
    note = None if model == _MODELS[0] else f"Job analysis ran on {model}."
    return profile, note


def _heuristic_profile(job_title: str) -> RoleProfile:
    """Last resort so a run still returns something useful if the model fails."""
    title = (job_title or "the role").strip()
    generic = [
        ("R01", "Directly relevant experience in this function", "experience", "must"),
        ("R02", "Core technical or functional skills the role runs on", "skill", "must"),
        ("R03", "Evidence of delivering measurable outcomes", "experience", "strong"),
        ("R04", "Seniority and scope appropriate to the role", "experience", "strong"),
        ("R05", "Communication and stakeholder handling", "behavioural", "strong"),
        ("R06", "Relevant domain or industry context", "domain", "nice"),
    ]
    return RoleProfile(
        title=title,
        summary=f"No usable job description was analysed, so {title} is being assessed against a generic rubric for its title. Paste the real posting for a far sharper result.",
        requirements=[Requirement(id=i, text=t, category=c, criticality=k, proof="Concrete examples in the CV.") for i, t, c, k in generic],
        dimension_weights=dict(_FALLBACK_WEIGHTS),
    )
