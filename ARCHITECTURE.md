# Sabha — how it works

This document explains the whole system: the idea behind it, what happens on a
single request, and what every file in the repository does and why.

It is written to be read start to finish by someone who has never seen the code.

---

## 1. What this is, in one paragraph

Sabha takes a job description and a CV and returns two numbers — a **score out
of 100** (how strong this CV is *for this role*) and a **match percentage**
(how much of what the job needs the candidate can actually evidence) — plus the
specific reasons for both and a prioritised list of fixes. It does this by
running seven AI assessors, each deliberately biased in a different direction,
across four independently-trained open-weight models, on the machine serving
the page. No CV is written to disk and none is sent to a third party.

---

## 2. The central idea: why this is not keyword matching

Keyword matching fails in both directions. It rejects people who can obviously
do the job but describe it in different words, and it passes people who simply
listed the right nouns. Everything about the ordering of this pipeline exists
to make keyword matching structurally impossible.

Three decisions carry that weight:

**The job is decomposed before the CV is read at all.** A model turns the
posting into 8–16 atomic requirements, and for each one it must commit *in
advance* to two things: what evidence would prove it (`proof`) and what
adjacent experience legitimately counts instead (`proxies`). Because this
happens before the candidate is visible, the rubric cannot be quietly bent to
flatter or punish the specific person being assessed. The same step decides how
much this role weights each scoring axis, so an executive hire and a graduate
role are not judged on the same things.

**Evidence indexing is plain Python, not a model.** Every later stage cites
evidence by id (`[E07]`). If a model produced the evidence index, it could
invent a unit and then "cite" it, and the citation check would wave it through.
Segmenting mechanically means an assessor can only ever point at text the
candidate actually wrote, and any id it invents fails validation and is
dropped. This is the integrity guarantee the whole system rests on, and it is
tested (`test_council.py` asserts every unit is a verbatim substring of the CV).

**Requirements are judged on capability, not vocabulary.** Each requirement is
classified `direct` / `transferable` / `partial` / `weak` / `absent` by two
independently-trained models, which must cite what decided it. `transferable`
is the whole point: *has not done this exact thing, but has demonstrably done
something that carries over*. In testing, a CV that never said "Kubernetes" but
described running containerised services at scale on ECS was correctly read as
`transferable`, not `absent` — and a BEng was accepted for "CS degree or
equivalent".

---

## 3. What happens on one request, end to end

```
 Browser                     server.py                pipeline.py            Ollama
    │                            │                         │                   │
    │─ POST /api/analyze ───────▶│                         │                   │
    │   (multipart: JD + CVs)    │ extract.py: file→text    │                   │
    │                            │ rate limit, size caps   │                   │
    │◀── {run_id} ───────────────│ store in memory only    │                   │
    │                            │                         │                   │
    │─ GET /api/stream/{id} ────▶│─ run() ────────────────▶│                   │
    │                            │                         │                   │
    │                            │   1. jd.analyse_job ────┼──────────────────▶│
    │◀═ SSE progress ════════════╪═════════════════════════│   requirements    │
    │                            │   2. evidence.index_cv  │   (no model)      │
    │                            │      atsaudit.audit     │   (no model)      │
    │◀═ SSE progress ════════════╪═════════════════════════│                   │
    │                            │   3. match.match_… ─────┼──────────────────▶│
    │◀═ SSE progress ════════════╪═════════════════════════│   2 models pooled │
    │                            │   4. council (7) ───────┼──────────────────▶│
    │◀═ SSE progress ════════════╪═════════════════════════│   grouped by model│
    │                            │   5. aggregate + synth ─┼──────────────────▶│
    │◀═ SSE result (JSON) ═══════│                         │                   │
    │                            │ delete run, drop CV     │                   │
```

Step by step:

1. **Job decomposition** (`jd.analyse_job`) — the posting becomes a
   `RoleProfile`: title, seniority, a plain-language summary of what the work
   actually demands, 8–16 `Requirement` objects each with criticality/proof/
   proxies, and the six `dimension_weights` for this role. No CV involved.
2. **CV indexing** (`evidence.index_cv`) — the CV becomes a list of
   `EvidenceUnit`s with ids `E01…`, each carrying its employer/period context.
   In parallel, `atsaudit.audit` runs a dozen mechanical checks in pure Python.
3. **Requirement matching** (`match.match_requirements`) — requirements are
   chunked six at a time and judged by two different models; the two verdicts
   are pooled by confidence, and disagreement *lowers* confidence rather than
   being hidden.
4. **The council** (`pipeline._run_member` × 7) — each member sees the role, the
   evidence block, the requirement verdicts and the CV(s) it is entitled to
   read, and returns six dimension scores plus strengths, concerns and a
   recommendation.
5. **Aggregation and coaching** (`aggregate.*`, `pipeline._synthesise`) — bias
   offsets are subtracted, outliers are clamped, dimensions are pooled, the two
   headline numbers are computed, and a final model call writes the
   improvements.

The CV text exists only inside the run record in memory and is set to `None` the
moment the run ends.

---

## 4. Repository map

| Path | Role |
|---|---|
| `council/config.py` | All runtime settings, every one overridable by env var |
| `council/schema.py` | The data contract every stage produces and consumes |
| `council/providers.py` | Async Ollama client + tolerant JSON parsing |
| `council/personas.py` | The seven assessors: biases, weight vectors, prompts |
| `council/jd.py` | Stage 1 — job → structured requirements |
| `council/extract.py` | PDF/DOCX/text → plain text |
| `council/evidence.py` | Stage 2a — CV → citable evidence units (no model) |
| `council/atsaudit.py` | Stage 2b — deterministic ATS checks (no model) |
| `council/match.py` | Stage 3 — requirement × evidence → verdicts |
| `council/aggregate.py` | Stage 5a — the scoring mathematics |
| `council/pipeline.py` | Orchestrates stages 1–5, runs the council |
| `council/server.py` | FastAPI app, SSE streaming, public-deployment guards |
| `council/static/*` | The single-page front end |
| `deploy/*` | launchd service + install script |
| `test_council.py` | 57 checks over every deterministic component |

---

## 5. File-by-file

### `council/__init__.py`
Package marker. Holds `__version__` and a docstring stating the design goal
(that nothing compares job strings to CV strings).

### `council/config.py`
A single frozen-ish `Settings` dataclass, instantiated once as `settings`.
Every field reads an environment variable prefixed `COUNCIL_`, so the launchd
plist configures a deployment without touching code.

| Setting | Default | Why |
|---|---|---|
| `ollama_host` | `http://127.0.0.1:11434` | Local daemon |
| `concurrency` | `3` | Members in flight at once. 4 exceeded usable VRAM and killed members with timeouts |
| `request_timeout_s` | `300` | General model call ceiling |
| `member_timeout_s` | `150` | Council members generate ≤1100 tokens; beyond this a model has stopped responding, not started thinking |
| `max_chars_cv` / `max_chars_jd` | 60k / 30k | Public endpoint, bounded input |
| `max_upload_bytes` | 8 MB | Public endpoint |
| `rate_limit_per_hour` | `12` | Per IP; each run costs real GPU time |
| `port` | `8700` | Loopback only; the tunnel makes it public |

### `council/schema.py`
The typed contract between stages, all Pydantic models. Defining these
explicitly is what lets a later stage cite an earlier stage's evidence by id
instead of re-reading free text and guessing.

Key types: `Requirement` (with `proof` and `proxies` — the anti-keyword
fields), `RoleProfile`, `EvidenceUnit`, `RequirementVerdict`, `MemberScore`,
`Improvement`, `ATSAudit`, and `CouncilResult` which is what the API returns.

It also holds the two weighting tables the scoring depends on:

```python
CRITICALITY_WEIGHT = {"must": 3.0, "strong": 2.0, "nice": 1.0}
COVERAGE_CREDIT    = {"direct": 1.0, "transferable": 0.75,
                      "partial": 0.5, "weak": 0.25, "absent": 0.0}
DIMENSIONS = ("relevance", "depth", "impact",
              "trajectory", "communication", "risk")
```

### `council/providers.py`
The only file that talks to a model, and it only ever talks to a **local**
Ollama daemon — there is deliberately no hosted-API path, because CVs are other
people's personal data on a public URL.

Two things matter here:

`parse_json(text)` salvages JSON from whatever a small model actually produced.
Small open-weight models wrap JSON in prose or fences even when told not to.
Rather than fail a whole council because one member wrote "Here is my
analysis:", it tries progressively looser reads: direct parse → fenced-block
extraction → a balanced-delimiter scan that correctly ignores braces inside
strings and escaped quotes. Only if all three fail does it return `None`.

`Ollama` is the async client. `resolve()` matches a member's preferred model
list against what is installed, loosely (so `qwen2.5:14b` matches an installed
`qwen2.5:14b-instruct-q4_K_M`) and falls back rather than failing. `generate()`
posts to `/api/generate` with `keep_alive: "15m"` (otherwise a seven-model
council pays the model-load cost seven times) and `num_ctx: 8192` (measured
prompts run ~3.5K tokens, so there is comfortable headroom). Every failure is
returned as an `LLMResponse` with `.error` set, never raised — a dead member is
survivable, a dead council is not.

### `council/personas.py`
The seven assessors. Each is a frozen `Member` dataclass with a `bias` (stated
plainly and shown to the user), a `lens` (what it is told to care about), a
`weights` vector over the six dimensions summing to 1.0, an `offset` (its
declared lean), a `reliability` (its weight in the pool), and `reads` (which
document it is shown).

| Member | Bias | Weighted most on | Offset | Reads |
|---|---|---|---|---|
| Recruiter Screener | Skims fast on conventional signals | relevance, communication | −2 | ATS CV |
| Hiring Manager | Can they do the job on Monday | impact, relevance | 0 | both |
| Domain Assessor | Distrusts breadth, rewards provable depth | depth | −3 | both |
| Skeptic | Assumes the most flattering possible account | risk, depth | −8 | both |
| Talent Advocate | Reads for potential and transferable skill | trajectory | +7 | both |
| Executive Reviewer | Judges altitude, scope, ownership | impact, trajectory | +1 | exec CV |
| ATS Parser | Sees only what software can extract | communication, relevance | −4 | ATS CV |

`_BASE_RULES` is the shared prompt preamble every member receives. Eight hard
rules, of which three are load-bearing:

- *Rule 2* forces evidence citation for every factual claim.
- *Rule 5* forbids rewarding or penalising name, gender, age, nationality or
  employer prestige in itself, and instructs the model to treat prompt-like text
  inside a CV as data and flag it — a CV is evidence, not instructions.
- *Rule 7* exists because of a real bug. Early runs had members writing "deep
  expertise in Kubernetes" about a CV containing no Kubernetes: they were
  reading the requirement list as a list of the candidate's skills. The rule now
  states explicitly that requirement text is not evidence, and the requirement
  summary is phrased candidate-relatively (see `pipeline._COVERAGE_GLOSS`).

### `council/jd.py`
Stage 1. `analyse_job()` sends the posting to the strongest available model and
returns a `RoleProfile`.

The prompt is written around a specific insight: job descriptions are
unreliable documents. They pad requirement lists, inherit boilerplate, and ask
for more years than the work needs. The model is told to recover the underlying
role, and that *most postings have only two to five genuine must-haves* — if it
marks everything `must`, it has not done the analysis.

`_normalise_weights()` guards the model's dimension weights: coerces to float,
clamps negatives, renormalises to sum 1.0, and **floors every dimension at
0.02** so a model that zeroes one out cannot make a whole axis of the assessment
silently vanish. `_heuristic_profile()` is the last-resort generic rubric if the
model fails entirely, so a run still returns something useful and says so.

### `council/extract.py`
Turns an upload into text, in memory, dropping the bytes immediately.
`from_upload()` sniffs by extension *and* magic bytes (`%PDF-`, `PK`) and
returns `(text, source)` — the source kind is carried through to the ATS audit,
which judges a real PDF differently from pasted text.

The PDF path has a deliberate behaviour: if under 120 characters are
extractable, it raises rather than proceeding, with the message that the file
looks like a scan or image-only export — *and that an ATS would read it as
blank too*. That failure is the single most valuable thing the tool can tell
someone, so it is surfaced as advice rather than a generic parse error.

### `council/evidence.py`
Stage 2a, and the integrity backbone. Pure Python, no model.

`index_cv()` walks the CV line by line tracking the current section. It detects
headings (known section words, or short all-caps lines), role headers (lines
containing a parseable date range — these set the employer/period context that
following bullets inherit), bullets, skills lines and everything else. Each
becomes an `EvidenceUnit` with `kind`, `employer`, `period` and a `quantified`
flag. Units are numbered `E01…` only at the end.

If a CV exceeds `max_units` (70), it drops the *least informative* units first —
ranked by kind, then unquantified, then length — rather than truncating and
losing the end of someone's career.

`render()` produces the evidence block the models see; `valid_ids()` gives the
set that `match.py` validates citations against.

### `council/atsaudit.py`
Stage 2b. Twelve weighted checks, all pure Python, all reproducible.

The reasoning is explicit: an LLM asked "is this ATS-friendly?" will cheerfully
hallucinate a verdict, but most of what actually breaks a CV in a real tracking
system is mechanical and can simply be measured. Checks cover a parseable email
(weight 3), phone, profile links, standard section headings (3), machine-
readable date ranges (3), bullet usage, quantified-result density (3),
duty-listing openers like "Responsible for", multi-column/table layout (2.5),
length, exotic glyphs, and file format.

Each check returns `pass` / `warn` / `fail` plus a `detail` and a concrete
`fix`. The score is the weighted pass rate — `pass` earns full weight, `warn`
half, `fail` none. Same CV in, same number out, every time.

### `council/match.py`
Stage 3, and the highest-leverage step in the pipeline since it drives the
headline match percentage.

Requirements are chunked six at a time (`_CHUNK = 6`) and sent to **two**
different model families concurrently. The prompt states outright that this is
not keyword matching, with the canonical example in both directions: a CV that
never says "Kubernetes" but describes running containerised services at scale
*is* strong evidence; a CV that lists "Kubernetes" in a skills line and nowhere
else is *weak* evidence.

Two defences run on every returned verdict:

- Citations to evidence ids that do not exist are silently dropped.
- A `direct` or `transferable` verdict with **no** surviving valid citation is
  kept but its confidence is capped at 0.35 — an unsupported positive verdict
  must not be trusted at face value.

`_pool()` merges the two models' verdicts by confidence-weighted credit, maps
the result back to a label via `_credit_to_label()` midpoints, unions the
citations, and — when the two models disagreed — multiplies confidence by 0.72
and appends the dissenting reading to the reasoning so the disagreement is
visible rather than averaged away.

### `council/aggregate.py`
Stage 5a: all the scoring mathematics, and no model calls.

**Bias correction.** `corrected_scores()` subtracts each member's declared
`offset` before anything is pooled. The Skeptic cannot drag the panel down by
being the Skeptic; the Advocate cannot lift it by being the Advocate. What
survives correction is the part of their judgement that is about the candidate.
(Test: members scoring 42 and 57 both correct to exactly 50.)

**Outlier damping.** `_fences()` computes Tukey fences (Q1 − 1.5·IQR,
Q3 + 1.5·IQR) per dimension; values outside are clamped to the fence. One member
occasionally returns a wild score on one axis — a 20 for risk against a panel
sitting between 70 and 95 — and on a heavily weighted axis that would visibly
move the headline. Clamping keeps the dissent (the value stays lowest) without
letting a model slip dominate. The number clamped is reported in the run notes.

**Competence weighting.** `pool_dimensions()` weights each member on each
dimension by `reliability × that member's own weight on that dimension`. The
ATS Parser therefore dominates machine-readability and barely touches impact,
which is right: a member should carry weight where it was designed to look.

**The score.** `overall_score()` = Σ(pooled dimension × the *role's* weight for
that dimension), normalised. The weights come from the job analysis, so the axes
are set by the job rather than fixed.

**The match percentage.** `match_percentage()`:

```
credit_r = COVERAGE_CREDIT[coverage]          # direct 1.0 … absent 0.0
if confidence < 0.5: credit_r *= (0.5 + confidence)   # unsure ⇒ pulled to neutral
weight_r = CRITICALITY_WEIGHT[criticality]    # must 3, strong 2, nice 1
raw      = 100 × Σ(credit_r × weight_r) / Σ(weight_r)
```

Then **the gate**, which is the part that makes this behave like real hiring:
one unevidenced must-have caps the result at **65%**, two or more cap it at
**45% − 3×(n−2)**. Averaging alone would let four satisfied nice-to-haves paper
over a missing must-have. The uncapped figure and the reason for the cap are
both returned and shown, e.g. *"coverage was 73% but is capped at 65% because a
must-have is unevidenced."*

**Consensus and confidence.** `consensus()` is the standard deviation of
bias-corrected member overalls, mapped so ~18 points of spread reads as total
disagreement. Spread that survives bias correction is genuine disagreement about
the candidate, so it is the honest input to `confidence_label()`, which
combines consensus (45%), mean verdict confidence (35%) and the fraction of the
panel that returned successfully (20%).

`verdict_label()` maps score + match + missing must-haves to *strong fit* /
*worth interviewing* / *borderline* / *not a fit yet*, with two or more missing
must-haves overriding any score.

### `council/pipeline.py`
The orchestrator, and the file with the most hard-won detail.

`_req_summary()` renders the requirement verdicts for the council. Each line is
phrased candidate-relatively via `_COVERAGE_GLOSS` — "NOT DIRECTLY EVIDENCED —
the candidate has NOT done this specific thing, but adjacent experience may
carry over" rather than a bare `TRANSFERABLE` label. Smaller models read
"Kubernetes → TRANSFERABLE" as "the candidate knows Kubernetes" and then repeat
it as fact; spelling it out leaves no room for that. The `Basis:` text is
truncated to 170 characters because every member pays prompt-processing time
for the whole block.

`_run_member()` builds one member's prompt and parses its response. It picks the
document the member is entitled to read, and never hands a member a blank one —
if only an executive CV was supplied, the ATS readers fall back to it, and vice
versa. Scores are coerced and clamped; anything unparseable yields a member with
`error` set rather than fabricated scores.

`run()` executes the five stages, and contains three pieces of reliability
machinery that exist because of a specific failure:

- **The council runs grouped by model**, busiest group first. Running all seven
  members freely meant Ollama had to evict a model that was still generating to
  load the next member's; the displaced requests then sat in the queue until
  they timed out, losing whole members from the panel.
- **Members fail fast** at `member_timeout_s` (150s) rather than the general
  300s, leaving time to recover.
- **Failed members are rescued** — retried once on whichever model has already
  answered successfully in *this* run. A panel silently missing its Skeptic or
  its Advocate is not the panel this tool claims to run, because bias correction
  assumes both poles are present. The substitution is recorded in the run notes,
  never hidden.

`_synthesise()` writes the improvements. Its prompt forbids suggesting the
candidate claim experience they lack, and specifically forbids phrasing that
*assumes* experience the evidence does not show — "add a bullet about your
Terraform project" is exactly that mistake when there is no Terraform project.
For a genuine gap it must either (a) name the adjacent experience they do have
and show how to surface it honestly, or (b) say plainly that this is a real gap
the CV cannot fix. Mechanical ATS failures the coach did not mention are
appended from `_deterministic_improvements()`, because those are reproducible
facts and should not depend on a model remembering them.

### `council/server.py`
The FastAPI application.

Endpoints: `GET /` (page), `/app.js`, `/styles.css`, `GET /api/health`,
`GET /api/council` (publishes each member's bias, weights and offset — the
panel is documented, not hidden), `POST /api/analyze`, `GET /api/stream/{id}`.

The two-call shape exists because SSE over POST is awkward: `/api/analyze`
validates, stores the run in memory and returns a `run_id`; `/api/stream/{id}`
opens the event stream and *drives* the run, emitting `progress` events, then
one `result` or `error`, then `done`.

Public-deployment guards, all present because the site is reachable by anyone
with the link: a per-IP hourly rate limit, upload and character caps, a
`Semaphore(2)` ceiling on concurrent runs so one visitor cannot monopolise the
GPU, security headers including a strict CSP on every response, and a
background reaper that sweeps abandoned runs after 15 minutes and prunes the
rate-limit table so it cannot grow one entry per visiting IP forever.

Storage policy is enforced in code: the run record holds the CV text, the
generator pops the record in its `finally` block, and `_drive()` sets
`rec["args"] = None` when the run ends. A 20-second keepalive comment keeps
proxies from dropping an idle stream.

### `council/static/index.html`
The single page. The header is a broadsheet masthead: a heavy/hairline rule
pair, a letterspaced kicker, the wordmark, the Devanagari सभा, and an italic
gloss of what the word means. Notable detail: an inline script runs *before* the stylesheet
link and redirects to add a trailing slash if missing. Served under a path
prefix, `styles.css` on `/council` resolves to `/styles.css` — a different
service entirely — so the page would load unstyled. Normalising first prevents
the browser from ever requesting the wrong asset. The footer states the design,
the bias handling, the privacy position and the limits in plain language.

### `council/static/styles.css`
One stylesheet, no framework, no external fonts. The CSP forbids remote
resources, and that constraint is deliberate: a hiring tool that leaks the
visitor's IP to a font CDN while promising their CV never leaves the machine
would be telling two different stories. So the typography is built entirely
from faces that ship with the operating system, chosen as a deliberate pairing
rather than left to the browser default:

| Token | Stack | Used for |
|---|---|---|
| `--font-masthead` | Didot → Bodoni 72 → Hoefler Text → Baskerville → Georgia | The wordmark and the two score gauges — large sizes only, because a high-contrast modern serif has hairline strokes that break up small |
| `--font-display` | Iowan Old Style → Charter → Palatino → Georgia | Headings, verdicts, member names, body prose — a transitional serif that stays robust at mid sizes |
| `--font-body` | system sans (`-apple-system` …) | Forms, dense metadata, small labels, where a sans is genuinely more legible |
| `--mono` | system mono | Evidence ids, model names, rewritten-bullet examples |

The design is broadsheet-editorial, which suits a tool whose output is a
considered verdict: a warm newsprint palette (paper is never grey), oxblood
rather than a saturated brand colour, 3px corners instead of pills, hairline
rules doing the work that borders and drop shadows usually do, and
letterspaced uppercase section labels. `font-variant-numeric: tabular-nums` is
set globally so columns of scores line up. There is a full dark-mode block
under `prefers-color-scheme`, and the layout collapses to a single column —
including the two-column colophon footer — below 760px.

`--sans` is kept as an alias because `app.js` references it inside generated
inline SVG.

### `council/static/app.js`
The client. Computes `BASE` from `location.pathname` so every request works
under any path prefix. Submits the form, opens an `EventSource`, appends
timestamped progress lines and lights up a chip per member, then renders the
result: two ring gauges drawn as inline SVG (no chart library), the verdict
badges, the role the council inferred, a requirement-by-requirement breakdown
where each cited evidence id carries the actual CV text as a tooltip, dimension
bars annotated with the role's own weights, the improvements, the panel with
each member's raw scores and published bias, and the ATS checklist. All
interpolated values pass through `esc()`.

### `deploy/com.hiringcouncil.server.plist`
The launchd agent. `RunAtLoad` + `KeepAlive` mean it starts at login and
restarts on crash. It binds uvicorn to `127.0.0.1` on purpose — the tunnel is
what makes the service public, and binding `0.0.0.0` would additionally expose
it to the whole local network. One worker, because the council saturates the GPU
on a single run and extra workers would queue behind each other while
multiplying resident model memory.

### `deploy/install.sh`
Idempotent installer: copies the plist, bootstraps the service, waits for
`/api/health`, then publishes the path on the tunnel. It also checks whether
Ollama is registered as a login item and warns if not — otherwise the site
returns errors after a reboot.

### `test_council.py`
57 checks over every deterministic component, run with `python3 test_council.py`
(no pytest dependency). Covers JSON salvage from malformed model output, ATS
discrimination between a good and a bad CV, evidence indexing — including the
assertion that **every unit is a verbatim substring of the CV** — bias
correction convergence, outlier fences, the match gate in all its branches,
verdict/confidence labels, and weight normalisation.

The model-driven stages can only be judged by reading their output, but
everything that turns their output into a number is ordinary code, and those
are the parts that decide what the candidate is actually told.

---

## 6. Configuration reference

All environment variables are prefixed `COUNCIL_`:

`OLLAMA_HOST`, `CONCURRENCY`, `TIMEOUT`, `MEMBER_TIMEOUT`, `MAX_CHARS_CV`,
`MAX_CHARS_JD`, `MAX_UPLOAD_BYTES`, `RATE_LIMIT_PER_HOUR`, `ROOT_PATH`, `PORT`.

---

## 7. Known limitations

- **It is slow**: 5–7 minutes per analysis, inherent to seven local agents plus
  two-model matching. The page streams progress so the wait is legible.
- **The offsets are declared, not learned.** Nothing tells this system who
  actually got hired, so there is no ground truth to calibrate against. They
  encode the lean each role was designed to have and are tuning parameters.
- **The fairness instruction is a mitigation, not a guarantee.** Members are
  told not to reward or penalise name, gender, age, nationality or employer
  prestige — but they are language models.
- **It is a drafting aid, not a hiring decision** and not a prediction of what
  any real employer will do.
- **It depends on the machine being on.** The service is served from a single
  Mac over a Tailscale Funnel.
- **A corrupt local model is hard to distinguish from GPU contention.** This was
  hit during development: `llama3.1:8b` stopped responding entirely and looked
  exactly like memory pressure. If members are substituted often, probe each
  model alone with a single small request and re-pull any that hang.
