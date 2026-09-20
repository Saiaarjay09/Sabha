"""Deterministic ATS checks — no model involved.

An LLM asked "is this ATS-friendly?" will cheerfully hallucinate a verdict.
Most of what actually breaks a CV in a real applicant tracking system is
mechanical and can simply be measured: is there a parseable email, do the date
ranges follow a consistent readable format, is the content trapped in a table.
Measuring it is both more accurate and more honest than asking a model, so
this entire module is plain Python and its results are reproducible.
"""

from __future__ import annotations

import re

from council.schema import ATSAudit

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d[\d\s.-]{7,}\d")
URL = re.compile(r"(?:https?://|www\.|linkedin\.com/)\S+", re.I)

MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_RANGE = re.compile(
    rf"(?:{MONTH}\.?\s*\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})\s*(?:-|–|—|to|until)\s*"
    rf"(?:{MONTH}\.?\s*\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|present|current|now|date)",
    re.I,
)

SECTION_WORDS = {
    "experience": r"(?:work\s+)?experience|employment|career history|professional background",
    "education": r"education|academic|qualifications",
    "skills": r"skills|competenc|technical proficienc|core competencies",
}

# Numbers that carry meaning — money, percentages, multiples, magnitudes.
METRIC = re.compile(
    r"(?:[$€£₹]\s?\d|\d+\s?(?:%|percent|x\b|bps)|\b\d[\d,.]*\s?(?:k|m|bn|b|million|billion|thousand|crore|lakh)\b|\b\d[\d,]{2,}\b)",
    re.I,
)

WEAK_OPENERS = re.compile(
    r"^\s*(?:-|\*|•)?\s*(?:responsible for|worked on|helped|assisted|involved in|participated in|tasked with|duties includ)",
    re.I,
)


def _bullets(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^(?:-|\*|•|·|‣|◦|–|—)\s+", s) and len(s) > 12:
            out.append(s)
    return out


def _check(cid, label, ok, weight, detail, fix=""):
    return {
        "id": cid,
        "label": label,
        "status": "pass" if ok is True else ("warn" if ok == "warn" else "fail"),
        "weight": weight,
        "detail": detail,
        "fix": fix,
    }


def audit(text: str, source: str = "text") -> ATSAudit:
    """Run every mechanical check and return a weighted score out of 100."""
    checks: list[dict] = []
    words = text.split()
    wc = len(words)
    lines = [l for l in text.split("\n") if l.strip()]
    lower = text.lower()

    # --- contact block -----------------------------------------------------
    has_email = bool(EMAIL.search(text))
    checks.append(_check(
        "email", "Email address is findable", has_email, 3.0,
        "Found a parseable email address." if has_email else "No email address the parser could find.",
        "" if has_email else "Put a plain-text email in the top few lines — not in a header, footer, or image.",
    ))

    has_phone = bool(PHONE.search(text))
    checks.append(_check(
        "phone", "Phone number is findable", has_phone or "warn", 1.5,
        "Found a phone number." if has_phone else "No phone number detected.",
        "" if has_phone else "Add a phone number in plain text with country code.",
    ))

    has_url = bool(URL.search(text))
    checks.append(_check(
        "links", "LinkedIn or portfolio link present", has_url or "warn", 1.0,
        "Found at least one profile or portfolio link." if has_url else "No LinkedIn/portfolio URL found.",
        "" if has_url else "Add your LinkedIn URL as visible text, not as a hyperlinked word.",
    ))

    # --- structure ---------------------------------------------------------
    found_sections = [
        name for name, pat in SECTION_WORDS.items()
        if re.search(rf"^\s*\W*{pat}\b", lower, re.M | re.I) or re.search(rf"\n\s*\W*{pat}\s*[:\n]", lower, re.I)
    ]
    n_sec = len(found_sections)
    checks.append(_check(
        "sections", "Standard section headings", True if n_sec >= 3 else ("warn" if n_sec == 2 else False), 3.0,
        f"Found {n_sec}/3 standard headings ({', '.join(found_sections) or 'none'}).",
        "" if n_sec >= 3 else "Use literal headings — Experience, Education, Skills. Creative headings don't map to ATS fields.",
    ))

    n_dates = len(DATE_RANGE.findall(text))
    checks.append(_check(
        "dates", "Parseable date ranges", True if n_dates >= 2 else ("warn" if n_dates == 1 else False), 3.0,
        f"Found {n_dates} machine-readable date range(s).",
        "" if n_dates >= 2 else "Write every role as 'Mar 2021 – Jun 2024'. Bare years or '21-24' often fail to parse.",
    ))

    bullets = _bullets(text)
    nb = len(bullets)
    checks.append(_check(
        "bullets", "Achievements in bullet form", True if nb >= 6 else ("warn" if nb >= 2 else False), 2.0,
        f"Found {nb} bullet point(s).",
        "" if nb >= 6 else "Break dense paragraphs into bullets — parsers and humans both skim them better.",
    ))

    # --- substance ---------------------------------------------------------
    metric_bullets = [b for b in bullets if METRIC.search(b)]
    ratio = (len(metric_bullets) / nb) if nb else 0.0
    checks.append(_check(
        "metrics", "Quantified results", True if ratio >= 0.4 else ("warn" if ratio >= 0.15 else False), 3.0,
        f"{len(metric_bullets)} of {nb} bullets carry a number ({ratio:.0%})." if nb
        else "No bullets found to measure.",
        "" if ratio >= 0.4 else "Attach a number to at least half your bullets — scale, %, money, time, or headcount.",
    ))

    # Check duty-listing openers on bullets *and* on ordinary body lines, or a
    # CV with no bullets at all would pass this check by default.
    candidates = bullets or [l.strip() for l in lines if len(l.strip()) > 12]
    weak = [b for b in candidates if WEAK_OPENERS.match(b)]
    checks.append(_check(
        "verbs", "Strong opening verbs", True if not weak else ("warn" if len(weak) <= 2 else False), 1.5,
        f"{len(weak)} line(s) open with a passive or duty-listing phrase." if weak
        else "Bullets open with active verbs.",
        "" if not weak else "Replace 'Responsible for / Worked on / Helped' with what you actually did: Built, Led, Cut, Shipped.",
    ))

    # --- formatting hazards ------------------------------------------------
    tabby = sum(1 for l in lines if "\t" in l or re.search(r"\S {3,}\S", l))
    tab_ratio = tabby / max(len(lines), 1)
    checks.append(_check(
        "layout", "No multi-column or table layout", True if tab_ratio < 0.15 else ("warn" if tab_ratio < 0.35 else False), 2.5,
        f"{tab_ratio:.0%} of lines look like columns or table rows.",
        "" if tab_ratio < 0.15 else "Move to a single-column layout. Tables and columns get read out of order or dropped entirely.",
    ))

    if wc == 0:
        length_ok, ldetail = False, "No text extracted."
    elif wc < 250:
        length_ok, ldetail = False, f"Only {wc} words — thin for a full CV."
    elif wc > 1400:
        length_ok, ldetail = "warn", f"{wc} words — long; the first screen may never reach your best material."
    else:
        length_ok, ldetail = True, f"{wc} words — a sensible length."
    checks.append(_check(
        "length", "Sensible length", length_ok, 1.5, ldetail,
        "" if length_ok is True else "Aim for roughly 400–1,000 words: two pages at most for senior roles, one earlier on.",
    ))

    glyphs = len(re.findall(r"[^\x00-\x7F–—‘’“”éèüöäçñ]", text))
    glyph_ratio = glyphs / max(wc, 1)
    checks.append(_check(
        "glyphs", "No exotic glyphs or icons", True if glyph_ratio < 0.05 else "warn", 1.0,
        f"{glyphs} unusual character(s) detected.",
        "" if glyph_ratio < 0.05 else "Strip icon fonts and decorative symbols — they arrive as mojibake or get dropped.",
    ))

    if source == "pdf":
        src_ok, sdetail, sfix = True, "Text-based PDF — extracted cleanly.", ""
    elif source == "docx":
        src_ok, sdetail, sfix = True, "Word document — widely parseable.", ""
    else:
        src_ok, sdetail, sfix = "warn", "Pasted as plain text, so original formatting wasn't assessed.", \
            "Upload the actual file you submit to check its real formatting."
    checks.append(_check("source", "File format", src_ok, 1.0, sdetail, sfix))

    earned = sum(c["weight"] * (1.0 if c["status"] == "pass" else 0.5 if c["status"] == "warn" else 0.0) for c in checks)
    total = sum(c["weight"] for c in checks)
    return ATSAudit(score=round(100 * earned / total, 1) if total else 0.0, checks=checks, parsed_chars=len(text))
