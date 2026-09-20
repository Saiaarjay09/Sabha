"""The data contract between stages.

Every stage of the pipeline consumes and produces one of these. Keeping them
explicit is what lets a later stage cite an earlier stage's evidence by id
instead of re-reading free text and guessing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Criticality = Literal["must", "strong", "nice"]
Coverage = Literal["direct", "transferable", "partial", "weak", "absent"]

# How much a requirement counts, by how badly the job needs it.
CRITICALITY_WEIGHT: dict[str, float] = {"must": 3.0, "strong": 2.0, "nice": 1.0}

# How much credit a candidate earns for a requirement, by quality of evidence.
# "transferable" is the whole point of the exercise: the candidate has not done
# this exact thing, but has done something that demonstrably carries over.
COVERAGE_CREDIT: dict[str, float] = {
    "direct": 1.0,
    "transferable": 0.75,
    "partial": 0.5,
    "weak": 0.25,
    "absent": 0.0,
}

DIMENSIONS: tuple[str, ...] = (
    "relevance",      # does this experience bear on THIS role
    "depth",          # is the claimed expertise credible and deep
    "impact",         # outcomes, scope, quantified results
    "trajectory",     # growth and direction, not just current position
    "communication",  # how well the CV itself argues its case
    "risk",           # 100 = no red flags, 0 = serious concerns
)

DIMENSION_LABELS: dict[str, str] = {
    "relevance": "Relevance to role",
    "depth": "Depth of expertise",
    "impact": "Impact & outcomes",
    "trajectory": "Career trajectory",
    "communication": "Communication",
    "risk": "Risk profile",
}


class Requirement(BaseModel):
    """One atomic thing the job actually needs.

    `proof` and `proxies` are what make the later matching dynamic: the model
    commits, before ever seeing the CV, to what would count as evidence and
    what adjacent experience it is willing to accept instead.
    """

    id: str
    text: str
    category: Literal["skill", "experience", "domain", "credential", "behavioural", "logistical"] = "skill"
    criticality: Criticality = "strong"
    proof: str = Field(default="", description="What evidence in a CV would prove this")
    proxies: list[str] = Field(default_factory=list, description="Adjacent experience that legitimately carries over")


class RoleProfile(BaseModel):
    """What the pipeline understood the job to be, before reading any CV."""

    title: str = ""
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "principal", "executive"] = "mid"
    role_family: str = ""
    summary: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    # Role-adaptive dimension weights. An executive hire is not judged on the
    # same axes as a graduate engineer, so the job decides the axes.
    dimension_weights: dict[str, float] = Field(default_factory=dict)
    unstated_expectations: list[str] = Field(default_factory=list)


class EvidenceUnit(BaseModel):
    """One citable claim lifted from the CV."""

    id: str
    text: str
    kind: Literal["role", "achievement", "skill", "education", "project", "other"] = "other"
    employer: str = ""
    period: str = ""
    quantified: bool = False


class RequirementVerdict(BaseModel):
    """The judgement on a single requirement, with its receipts."""

    requirement_id: str
    coverage: Coverage = "absent"
    confidence: float = 0.5
    evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""
    gap: str = Field(default="", description="What is missing, if anything")


class MemberScore(BaseModel):
    """One council member's verdict."""

    member: str
    model: str
    scores: dict[str, float] = Field(default_factory=dict)   # dimension -> 0..100, raw
    headline: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    recommendation: Literal["strong_yes", "yes", "borderline", "no"] = "borderline"
    latency_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.scores)


class Improvement(BaseModel):
    """One concrete, actionable fix."""

    priority: Literal["critical", "high", "medium"] = "medium"
    area: str = ""
    problem: str = ""
    action: str = ""
    example: str = ""


class ATSAudit(BaseModel):
    """Deterministic, non-LLM checks on the machine-readable CV."""

    score: float = 0.0
    checks: list[dict] = Field(default_factory=list)
    parsed_chars: int = 0


class CouncilResult(BaseModel):
    score: float = 0.0                 # 0..100 overall strength for THIS role
    match_pct: float = 0.0             # 0..100 requirement coverage
    verdict: str = "borderline"
    confidence: str = "medium"
    consensus: float = 0.0             # 0..100, how much the panel agreed
    role: RoleProfile = Field(default_factory=RoleProfile)
    dimensions: dict[str, float] = Field(default_factory=dict)
    members: list[MemberScore] = Field(default_factory=list)
    requirements: list[RequirementVerdict] = Field(default_factory=list)
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    improvements: list[Improvement] = Field(default_factory=list)
    ats: ATSAudit = Field(default_factory=ATSAudit)
    blocking_gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    elapsed_s: float = 0.0
    notes: list[str] = Field(default_factory=list)
