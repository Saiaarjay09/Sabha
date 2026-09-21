# Sabha

*Sabha* (सभा) is Sanskrit for a deliberative assembly — a council convened to
weigh a matter and reach a judgement.

A panel of seven AI assessors, each deliberately biased in a different
direction, reads a CV against a specific job and argues about it. You get a
score out of 100, a requirement-match percentage, and the specific reasons for
both.

Everything runs on local open-weight models. A submitted CV never leaves the
machine and is never written to disk.

## Try it

**→ https://haven.taila6d3cb.ts.net/council/**

Paste a job description and a CV. Nothing you paste is stored.

Runtime varies far more than a single figure suggests. Measured runs that
completed with the full panel have ranged from **6 to 11 minutes**, and a run
that loses a member to a timeout and retries it can pass fifteen. The variance
is dominated by what the GPU already has resident, not by the length of your
CV. The page streams each assessor's progress so the wait is legible rather
than a blank spinner.

It is served from a single Mac over a Tailscale Funnel rather than cloud
infrastructure, so if the link does not resolve, that machine is asleep — it
is not an outage. Run it yourself with the instructions below.

## What makes the matching dynamic

The thing this is built to avoid is keyword matching, which fails in both
directions: it rejects people who can obviously do the job but describe it in
different words, and it passes people who listed the right nouns.

So the pipeline never compares the job text to the CV text. It runs in an order
designed to make that impossible:

1. **The job is decomposed first, before the CV is read at all.** A model turns
   the posting into 8–16 atomic requirements. For each one it must commit, in
   advance, to two things: what evidence would *prove* it, and which adjacent
   experience it is willing to accept as a *proxy*. Because this happens before
   the candidate is visible, the rubric cannot be quietly bent to flatter or
   punish the person in front of it. The same step also decides how much this
   particular role weights each scoring axis — an executive hire and a graduate
   role are not judged on the same things.

2. **The CV is indexed mechanically into citable evidence units.** This is
   plain Python, not a model, and that is deliberate: every later stage cites
   evidence by id, so if a model produced the index it could invent a unit and
   then "cite" it. Segmenting in code means an assessor can only ever point at
   text the candidate actually wrote, and any id it invents fails validation
   and is dropped.

3. **Each requirement is judged against that evidence by two independently
   trained models**, which must classify it as `direct`, `transferable`,
   `partial`, `weak` or `absent` and cite what decided it. `transferable` is
   the whole point: *has not done this exact thing, but has demonstrably done
   something that carries over*. The two verdicts are pooled by confidence, and
   disagreement between them lowers the confidence rather than being hidden.

4. **Then the council convenes**, seeing the role, the evidence and the
   requirement verdicts.

5. **Aggregation, then coaching.**

## The council

| Member | Bias | Weighted most on |
|---|---|---|
| Recruiter Screener | Skims fast on conventional signals; unforgiving of a slow case | relevance, communication |
| Hiring Manager | Only asks whether they can do the job on Monday | impact, relevance |
| Domain Assessor | Distrusts breadth, rewards provable depth | depth |
| Skeptic | Assumes the CV is the most flattering possible account | risk, depth |
| Talent Advocate | Reads for potential and transferable capability | trajectory |
| Executive Reviewer | Judges altitude: scope, ownership, stakeholders | impact, trajectory |
| ATS Parser | Sees only what software can extract; infers nothing | communication, relevance |

Members are spread across four independently-trained open-weight families —
Qwen, Llama, Mistral and Gemma — because running seven personas on one base
model produces seven correlated opinions wearing different hats, and no amount
of prompting fixes that.

Four rather than more is a memory trade. Ollama reports 37.4 GiB of usable
VRAM on the machine this was built on, and these four sit at roughly 31 GiB
resident together, so they stay loaded across a run. A fifth large model
(phi4, 11 GiB) pushes past that, so phi is kept in the fallback lists only.

Three mechanisms keep a run from losing members, and they exist because of a
failure worth recording. Mid-development, `llama3.1:8b` stopped responding on
this machine entirely — every request timed out, in JSON mode and in free-text
mode, sequential and concurrent alike. It looked exactly like GPU contention,
and it was misdiagnosed twice as a concurrency problem before an isolated probe
of one model at a time showed the model itself was simply dead; the local copy
was corrupt and `ollama rm && ollama pull` fixed it. Meanwhile the stuck
requests held the GPU and starved the other models' members, so a single
corrupt model was quietly taking three of seven assessors off the panel.

The mitigations that came out of that:

- **The council runs grouped by model**, one group at a time. Every member runs
  against weights already resident, and the next model is only loaded once the
  previous group has finished with the GPU.
- **Members fail fast** (150s, vs 300s for other calls). A member generates at
  most ~1100 tokens, so anything beyond that is a model that has stopped
  responding, not one that is thinking — and failing fast leaves time to retry.
- **Failed members are rescued**, re-run once on a model that has already
  answered successfully in that same run. A panel silently missing its Skeptic
  or its Advocate is not the panel this tool claims to run, because bias
  correction assumes both poles are present. The substitution is recorded in
  the run notes rather than hidden.

If members do get substituted often, suspect the model before the code: probe
each one alone with a single small request, and re-pull any that hang.

On a machine with less memory, cut the council rather than letting it thrash:
point several members at the same smaller model in `council/personas.py`.

### How bias is handled

Bias is the feature; uncorrected bias would be the bug. Three mechanisms:

- **Declared offsets.** Every member has a known lean (the Skeptic −8, the
  Advocate +7) which is *subtracted* before anything is pooled. The Skeptic
  cannot drag the panel down by being the Skeptic. What survives correction is
  the part of its judgement that is about the candidate.
- **Competence weighting.** Each dimension is pooled using each member's own
  weight on that dimension. The ATS Parser dominates machine-readability and
  barely touches impact, which is right — a member should carry weight where it
  was designed to look.
- **Disagreement is reported, not hidden.** Spread that survives bias
  correction is real disagreement, and it sets the confidence label.

**These offsets are declared tuning parameters, not calibrations learned from
outcomes.** Nobody tells this system who actually got hired, so there is no
ground truth to calibrate against. They encode the lean each role was designed
to have. Tune them in `council/personas.py`.

## The two numbers

**Score /100** — CV strength *for this role*, pooled across six axes weighted
by what the job analysis said this role needs.

**Match %** — requirement coverage: each requirement's credit (direct 1.0,
transferable 0.75, partial 0.5, weak 0.25, absent 0) times how badly the job
needs it (must 3, strong 2, nice 1). Low-confidence verdicts are pulled toward
neutral rather than counted at face value.

Match is **gated, not just averaged**. Four satisfied nice-to-haves should not
paper over a missing must-have, so one unevidenced must-have caps the headline
at 65%, and two or more cap it at 45% — with the reason reported. Real hiring
works this way and an unweighted average does not.

The ATS score is separate and is **not** a model opinion: it is a dozen
mechanical checks (parseable email, consistent date ranges, table layout,
metric density, duty-listing verbs) run in plain code. Same CV, same result,
every time.

For a full code tour — the design, an end-to-end walkthrough of one request,
and what every file does — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Running it

Needs [Ollama](https://ollama.com) and Python 3.11+.

```bash
pip3 install -r requirements.txt
for m in qwen2.5:14b llama3.1:8b mistral-nemo:12b gemma2:9b; do ollama pull $m; done
python3 -m uvicorn council.server:app --host 127.0.0.1 --port 8700
```

Then open http://127.0.0.1:8700. The council degrades gracefully if only some
models are installed — members fall back to what is available and the
substitution is recorded in the run notes.

### Permanent deployment

`deploy/install.sh` installs a launchd agent (starts at login, restarts on
crash) and publishes the app on an existing Tailscale Funnel hostname under
`/council`. Re-running it is safe.

```bash
./deploy/install.sh
```

Ollama must also start at login, or the site returns an error after a reboot —
enable "Launch at login" in the Ollama app. The install script warns if it
doesn't look set.

## Privacy and limits

The server binds to loopback; Tailscale Funnel is what makes it public. A CV is
held in memory for the life of its run and dropped when the result is
delivered — never written to disk, never sent to a third party. Abandoned runs
are reaped after 15 minutes. There is a per-IP rate limit and a cap on
concurrent runs, because the machine serving this is a laptop.

This is a drafting aid, not a hiring decision and not a prediction of what any
real employer will do. The assessors are instructed never to reward or penalise
name, gender, age, nationality, or the prestige of a school or employer in
itself — but they are language models, and that instruction is a mitigation,
not a guarantee. Treat the score as a structured second opinion.

A CV is treated as data, never as instructions: assessors are told to ignore
any prompt-like text inside a CV and to flag it as a concern.
