"""Index the CV into citable units.

This is deliberately *not* done by a model. Every later stage cites evidence by
id, and the whole integrity argument of the system rests on those citations
pointing at text the candidate actually wrote. If a model produced the evidence
index it could invent a unit and then "cite" it, and the citation check would
wave it through. Segmenting mechanically means a member can only ever point at
real lines of the real CV — and any id it invents fails validation loudly.
"""

from __future__ import annotations

import re

from council.atsaudit import DATE_RANGE, METRIC
from council.schema import EvidenceUnit

_BULLET = re.compile(r"^(?:-|\*|•|·|‣|◦|–|—|\d+[.)])\s+")
_HEADING = re.compile(
    r"^\s*\W{0,3}(experience|employment|career|education|academic|skills|competenc|"
    r"projects?|certification|summary|profile|objective|publications?|awards?|interests?|"
    r"languages?|volunteer|references?)\b",
    re.I,
)
_SKILLS_HEAD = re.compile(r"skills|competenc|technical|tools|languages", re.I)
_EDU_HEAD = re.compile(r"education|academic|qualification|certification", re.I)
_PROJ_HEAD = re.compile(r"project", re.I)

# "Senior Engineer, Monzo — Mar 2021 – Jun 2024" style lines.
_SEPARATORS = re.compile(r"\s+[—–\-|·,]\s+|\t+")


def _looks_like_heading(line: str) -> bool:
    s = line.strip().rstrip(":")
    if not s or len(s) > 60:
        return False
    if _HEADING.match(s):
        return True
    # A short ALL-CAPS line with no sentence punctuation is almost always a heading.
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(s.split()) <= 5


def _employer_and_period(line: str) -> tuple[str, str]:
    m = DATE_RANGE.search(line)
    period = m.group(0).strip() if m else ""
    head = line[: m.start()] if m else line
    parts = [p.strip(" ,|–—-\t") for p in _SEPARATORS.split(head) if p.strip(" ,|–—-\t")]
    employer = parts[-1] if len(parts) > 1 else (parts[0] if parts else "")
    return employer[:80], period[:40]


def index_cv(text: str, max_units: int = 70) -> list[EvidenceUnit]:
    """Segment a CV into evidence units, preserving each unit's exact text."""
    units: list[EvidenceUnit] = []
    section = ""
    cur_employer = ""
    cur_period = ""

    raw_lines = [l.rstrip() for l in text.split("\n")]
    for raw in raw_lines:
        line = raw.strip()
        if len(line) < 3:
            continue

        if _looks_like_heading(line):
            section = line.strip().rstrip(":")
            # A new top-level section ends the previous employer's context.
            if not _PROJ_HEAD.search(section):
                cur_employer, cur_period = "", ""
            continue

        is_bullet = bool(_BULLET.match(line))
        body = _BULLET.sub("", line).strip() if is_bullet else line
        if len(body) < 8:
            continue

        has_dates = bool(DATE_RANGE.search(line))
        quantified = bool(METRIC.search(body))

        if has_dates and not is_bullet:
            # A role/education header: record it, and make it the context that
            # following bullets inherit.
            emp, per = _employer_and_period(line)
            cur_employer, cur_period = emp, per
            kind = "education" if _EDU_HEAD.search(section) else "role"
            units.append(EvidenceUnit(id="", text=body[:400], kind=kind, employer=emp, period=per, quantified=quantified))
            continue

        if _SKILLS_HEAD.search(section) and not is_bullet:
            kind = "skill"
        elif _EDU_HEAD.search(section):
            kind = "education"
        elif _PROJ_HEAD.search(section):
            kind = "project"
        elif is_bullet:
            kind = "achievement"
        else:
            kind = "other"

        units.append(EvidenceUnit(
            id="", text=body[:400], kind=kind,
            employer=cur_employer if kind in ("achievement", "project") else "",
            period=cur_period if kind in ("achievement", "project") else "",
            quantified=quantified,
        ))

    # If the CV is longer than the budget, drop the least informative units
    # first rather than truncating and losing the end of someone's career.
    if len(units) > max_units:
        rank = {"role": 0, "achievement": 1, "project": 2, "skill": 3, "education": 4, "other": 5}
        keep = sorted(
            sorted(range(len(units)), key=lambda i: (rank[units[i].kind], not units[i].quantified, -len(units[i].text)))[:max_units]
        )
        units = [units[i] for i in keep]

    for n, u in enumerate(units, 1):
        u.id = f"E{n:02d}"
    return units


def render(units: list[EvidenceUnit]) -> str:
    """The evidence block as the council sees it."""
    out = []
    for u in units:
        ctx = " · ".join(p for p in (u.employer, u.period) if p)
        out.append(f"[{u.id}] ({u.kind}{': ' + ctx if ctx else ''}) {u.text}")
    return "\n".join(out)


def valid_ids(units: list[EvidenceUnit]) -> set[str]:
    return {u.id for u in units}
