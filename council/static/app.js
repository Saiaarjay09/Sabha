/* Hiring Council — client.
   Served under a path prefix on a shared Tailscale hostname, so every URL is
   resolved relative to the current page rather than the domain root. */

const BASE = location.pathname.endsWith('/') ? location.pathname : location.pathname + '/';

const $ = (id) => document.getElementById(id);
const form = $('form'), go = $('go'), progress = $('progress'),
      logEl = $('log'), chipsEl = $('chips'), results = $('results');

const MEMBERS = [
  ['screener', 'Recruiter Screener'],
  ['hiring_manager', 'Hiring Manager'],
  ['technical', 'Domain Assessor'],
  ['skeptic', 'Skeptic'],
  ['advocate', 'Talent Advocate'],
  ['executive', 'Executive Reviewer'],
  ['ats_machine', 'ATS Parser'],
];

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let t0 = 0;
let lastResult = null;   // the benchmark a re-scan is compared against

function log(msg, isNow) {
  const el = document.createElement('div');
  const t = ((Date.now() - t0) / 1000).toFixed(0).padStart(3, ' ');
  el.innerHTML = `<span class="t">${t}s</span>  ${esc(msg)}`;
  if (isNow) el.className = 'now';
  logEl.querySelectorAll('.now').forEach(n => n.classList.remove('now'));
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

function resetChips() {
  chipsEl.innerHTML = MEMBERS
    .map(([n, label]) => `<span class="chip" data-m="${n}">${esc(label)}</span>`).join('');
}

const COVERAGE = {
  direct:       ['good',  'Direct'],
  transferable: ['info',  'Transferable'],
  partial:      ['warn',  'Partial'],
  weak:         ['warn',  'Weak'],
  absent:       ['bad',   'Not evidenced'],
};
const CRIT = { must: ['bad', 'Must have'], strong: ['warn', 'Strong'], nice: ['muted', 'Nice to have'] };
const REC = {
  strong_yes: ['good', 'Strong yes'], yes: ['good', 'Yes'],
  borderline: ['warn', 'Borderline'], no: ['bad', 'No'],
};

function band(v) { return v >= 70 ? 'good' : v >= 50 ? 'warn' : 'bad'; }

/* A ring gauge, drawn inline — no chart library, and it reads correctly in
   both themes because it inherits currentColor from its band class. */
function ring(value, size = 118) {
  const r = (size - 14) / 2, c = 2 * Math.PI * r, on = (Math.max(0, Math.min(100, value)) / 100) * c;
  const col = `var(--${band(value) === 'good' ? 'good' : band(value) === 'warn' ? 'warn' : 'bad'})`;
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${value} out of 100">
    <circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="var(--rule)" stroke-width="9"/>
    <circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="${col}" stroke-width="9"
      stroke-linecap="round" stroke-dasharray="${on.toFixed(1)} ${(c - on).toFixed(1)}"
      transform="rotate(-90 ${size/2} ${size/2})"/>
    <text x="50%" y="50%" text-anchor="middle" dy=".35em" font-size="27" font-weight="700"
      fill="var(--ink)" font-family="var(--font-masthead)">${Math.round(value)}</text>
  </svg>`;
}

function tag(kind, label) { return `<span class="tag badge ${kind}">${esc(label)}</span>`; }

function render(r) {
  const evById = Object.fromEntries((r.evidence || []).map(e => [e.id, e.text]));
  const reqById = Object.fromEntries((r.role.requirements || []).map(q => [q.id, q]));
  const vBand = ({ 'strong fit': 'good', 'worth interviewing': 'good', 'borderline': 'warn' })[r.verdict] || 'bad';

  const cites = (ids) => !ids || !ids.length ? '' :
    `<div class="cites">${ids.map(i => `<span title="${esc(evById[i] || 'not found')}">${esc(i)}</span>`).join('')}</div>`;

  /* headline */
  let h = `<div class="scores">
    <div class="score-card">
      <div class="label">CV strength for this role</div>
      ${ring(r.score)}
      <div class="sub">weighted across six axes the job itself set</div>
    </div>
    <div class="score-card">
      <div class="label">Requirement match</div>
      ${ring(r.match_pct)}
      <div class="sub">${esc(r.match_pct)}% of what this job needs, weighted by how badly it needs it</div>
    </div>
  </div>

  <div class="verdict-row">
    ${tag(vBand, r.verdict)}
    ${tag('muted', `confidence: ${r.confidence}`)}
    ${tag('muted', `panel agreement: ${Math.round(r.consensus)}%`)}
    ${tag(r.ats.compliant ? 'good' : 'bad', r.ats.compliant ? 'ATS compliant' : 'not ATS compliant')}
    ${tag(band(r.ats.score), `ATS readability: ${r.ats.score}`)}
    ${tag('muted', `${r.elapsed_s}s`)}
  </div>`;

  if (r.delta) {
    const d = r.delta;
    const arrow = (n) => n > 0 ? '▲' : n < 0 ? '▼' : '–';
    const dband = (n) => n > 0 ? 'good' : n < 0 ? 'bad' : 'muted';
    h += `<div class="card"><h2>What changed since the last scan</h2>
      <div class="verdict-row">
        ${tag(dband(d.score.change), `Score ${d.score.from} → ${d.score.to}  ${arrow(d.score.change)} ${Math.abs(d.score.change)}`)}
        ${tag(dband(d.match_pct.change), `Match ${d.match_pct.from}% → ${d.match_pct.to}%  ${arrow(d.match_pct.change)} ${Math.abs(d.match_pct.change)}`)}
        ${d.verdict.from !== d.verdict.to ? tag('info', `${d.verdict.from} → ${d.verdict.to}`) : tag('muted', `still ${esc(d.verdict.to)}`)}
      </div>`;
    if (d.requirements?.length) {
      h += `<p class="hint">The job was not re-analysed — the previous scan's requirements were
        reused, so these movements are comparable. Note that the models are not deterministic:
        a requirement can shift a step between runs without anything having changed, so treat
        small movements as noise and look at the ones you actually argued for.</p>`;
      for (const q of d.requirements) {
        h += `<div class="req"><div class="req-head">
            ${tag(q.improved ? 'good' : 'bad', `${esc(q.from)} → ${esc(q.to)}`)}
            <div class="req-text">${esc(q.text || q.id)}</div>
            ${q.from_claim ? tag('warn', 'on your word') : tag('muted', 'from the CV')}
          </div></div>`;
      }
    } else {
      h += `<p class="hint">No requirement changed its verdict.</p>`;
    }
    h += `</div>`;
  }

  if (r.summary) h += `<div class="card"><h2>Where you stand</h2><p class="sum">${esc(r.summary)}</p></div>`;

  if (r.blocking_gaps?.length) {
    h += `<div class="card"><h2>Blocking gaps</h2>
      <p class="hint">These are must-haves the council could not find evidence for. They cap your match percentage no matter how well the rest scored.</p>
      <ul>${r.blocking_gaps.map(g => `<li>${esc(g)}</li>`).join('')}</ul></div>`;
  }

  /* what the job was understood to be */
  h += `<div class="card"><h2>What the council understood the job to need</h2>
    <p><b>${esc(r.role.title)}</b> · ${esc(r.role.seniority)}${r.role.role_family ? ' · ' + esc(r.role.role_family) : ''}</p>
    <p class="req-why">${esc(r.role.summary)}</p>`;
  if (r.role.unstated_expectations?.length) {
    h += `<p class="ul-title">Unstated expectations it inferred</p>
      <ul style="margin:4px 0 0;font-size:13.5px;color:var(--ink-dim)">
      ${r.role.unstated_expectations.map(u => `<li>${esc(u)}</li>`).join('')}</ul>`;
  }
  h += `</div>`;

  /* requirements */
  h += `<div class="card"><h2>Requirement by requirement</h2>`;
  for (const v of r.requirements || []) {
    const q = reqById[v.requirement_id] || {};
    const [cb, cl] = COVERAGE[v.coverage] || COVERAGE.absent;
    const [kb, kl] = CRIT[q.criticality] || CRIT.strong;
    h += `<div class="req">
      <div class="req-head">
        ${tag(kb, kl)}
        <div class="req-text">${esc(q.text || v.requirement_id)}</div>
        ${tag(cb, cl)}
        ${v.from_claim ? tag('warn', 'unverified') : ''}
      </div>
      <p class="req-why">${esc(v.reasoning)}${v.gap ? ` <b>Gap:</b> ${esc(v.gap)}` : ''}</p>
      ${cites(v.evidence_ids)}
    </div>`;
  }
  h += `</div>`;

  /* dimensions */
  h += `<div class="card"><h2>Scores by dimension</h2>
    <p class="hint">Bold percentages are how much this specific job weighted each axis — set by the job analysis, not fixed.</p>`;
  const LBL = {
    relevance: 'Relevance to role', depth: 'Depth of expertise', impact: 'Impact &amp; outcomes',
    trajectory: 'Career trajectory', communication: 'Communication', risk: 'Risk profile',
  };
  for (const [k, label] of Object.entries(LBL)) {
    const val = r.dimensions[k] ?? 0, w = Math.round((r.role.dimension_weights?.[k] ?? 0) * 100);
    h += `<div class="bar-row">
      <div class="bar-label">${label} <b>${w}%</b></div>
      <div class="bar"><i style="width:${Math.max(1, val)}%;background:var(--${band(val) === 'good' ? 'good' : band(val) === 'warn' ? 'warn' : 'bad'})"></i></div>
      <div class="bar-val">${Math.round(val)}</div>
    </div>`;
  }
  h += `</div>`;

  /* improvements */
  if (r.improvements?.length) {
    h += `<div class="card"><h2>What to fix, in order</h2>`;
    for (const i of r.improvements) {
      const pb = i.priority === 'critical' ? 'bad' : i.priority === 'high' ? 'warn' : 'info';
      h += `<div class="imp ${esc(i.priority)}">
        <h4>${tag(pb, i.priority)} ${esc(i.area)}</h4>
        ${i.problem ? `<p class="problem">${esc(i.problem)}</p>` : ''}
        <p>${esc(i.action)}</p>
        ${i.example ? `<div class="example">${esc(i.example)}</div>` : ''}
      </div>`;
    }
    h += `</div>`;
  }

  /* the panel */
  h += `<div class="card"><h2>The panel</h2>
    <p class="hint">Each assessor's raw scores are shown. The headline number above is these, bias-corrected and pooled — each member weighted on the axes it was built to judge.</p>
    <div class="members">`;
  for (const m of r.members || []) {
    const meta = MEMBERS.find(x => x[0] === m.member);
    if (!m.scores || !Object.keys(m.scores).length) {
      h += `<div class="member"><h3>${esc(meta ? meta[1] : m.member)}</h3>
        <p class="bias">Did not return a usable assessment — excluded from the pool.</p>
        <div class="meta">${esc(m.error || '')}</div></div>`;
      continue;
    }
    const [rb, rl] = REC[m.recommendation] || REC.borderline;
    h += `<div class="member">
      <h3>${esc(meta ? meta[1] : m.member)} ${tag(rb, rl)}</h3>
      <p class="bias">${esc(m.bias_text || '')}</p>
      <p class="headline">${esc(m.headline)}</p>`;
    if (m.strengths?.length) h += `<div class="ul-title">Strengths</div><ul>${m.strengths.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
    if (m.concerns?.length) h += `<div class="ul-title">Concerns</div><ul>${m.concerns.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
    h += `<div class="meta">${esc(m.model)} · ${m.latency_s}s · ${Object.entries(m.scores).map(([k, v]) => k.slice(0, 3) + ' ' + Math.round(v)).join('  ')}</div>
    </div>`;
  }
  h += `</div></div>`;

  /* ats */
  h += `<div class="card"><h2>ATS checks — measured, not guessed</h2>
    <div class="imp ${r.ats.compliant ? 'medium' : 'critical'}" style="margin-bottom:16px">
      <h4>${tag(r.ats.compliant ? 'good' : 'bad', r.ats.compliant ? 'Compliant' : 'Not compliant')}
        ${r.ats.compliant ? 'This CV should survive automated parsing' : 'An automated parser would mangle or drop part of this CV'}</h4>
      ${r.ats.blockers?.length
        ? `<p class="problem">Blocking: ${r.ats.blockers.map(esc).join(' · ')}. These are failures a parser cannot recover from, so they decide compliance regardless of the score above.</p>`
        : `<p class="problem">No blocking failures. The score below still shows where it could read better.</p>`}
    </div>
    <p class="hint">These are mechanical checks run in plain code, not model opinions. They are reproducible: the same CV always produces the same result here. There is no published ATS standard to test against — this measures what is known to break real parsers.</p>`;
  for (const c of r.ats.checks || []) {
    const mark = c.status === 'pass' ? '✓' : c.status === 'warn' ? '!' : '✕';
    h += `<div class="check"><div class="mark ${esc(c.status)}">${mark}</div>
      <div class="body"><b>${esc(c.label)}</b> — ${esc(c.detail)}
      ${c.fix && c.status !== 'pass' ? `<div class="fix">→ ${esc(c.fix)}</div>` : ''}</div></div>`;
  }
  h += `<details class="raw"><summary>Evidence the council was given (${(r.evidence || []).length} units)</summary>
    <div style="margin-top:8px">${(r.evidence || []).map(e => `<div class="ev">[${esc(e.id)}] ${esc(e.text)}</div>`).join('')}</div>
  </details></div>`;

  h += `<div class="card"><h2>Reply to the council</h2>
    <p class="hint">Disagree, or left something off your CV? Say so and it will re-scan,
    reusing this scan's requirements as the benchmark so the two are comparable.
    Anything you add here is treated as <b>your word, not evidence</b> — it is cited
    separately, can never count as fully demonstrated, and is labelled in the result.</p>
    <div class="field">
      <textarea id="claims" placeholder="One point per line. For example:&#10;I did use Terraform at Ravelin — it is not on the CV because the project was internal.&#10;The migration was 400 services, not 240."></textarea>
    </div>
    <div class="actions">
      <button type="button" class="primary" id="rescan">Re-scan with this</button>
      <span class="note">Faster than the first run: the job is not re-analysed.</span>
    </div>
  </div>`;

  if (r.notes?.length) {
    h += `<div class="notes"><b>Run notes</b><ul>${r.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul></div>`;
  }

  results.innerHTML = h;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* Only the fields the delta needs go back to the server. The full result
   carries the evidence index, which is the CV text — there is no reason to
   send that back over the wire a second time to compute a comparison. */
function benchmark(r) {
  if (!r) return null;
  return {
    score: r.score,
    match_pct: r.match_pct,
    verdict: r.verdict,
    requirements: (r.requirements || []).map(v => ({ requirement_id: v.requirement_id, coverage: v.coverage })),
    role: {
      // The weights are part of the rubric, not decoration: a re-scan must be
      // judged on the same axes as the scan it is being compared against.
      dimension_weights: r.role?.dimension_weights || {},
      seniority: r.role?.seniority,
      title: r.role?.title,
      summary: r.role?.summary,
      requirements: (r.role?.requirements || []).map(q => ({
        id: q.id, text: q.text, criticality: q.criticality,
        category: q.category, proof: q.proof, proxies: q.proxies,
      })),
    },
  };
}

async function startRun(extra = {}) {
  go.disabled = true;
  go.textContent = 'Deliberating…';
  results.innerHTML = '';
  logEl.innerHTML = '';
  resetChips();
  progress.classList.add('on');
  progress.scrollIntoView({ behavior: 'smooth', block: 'center' });
  t0 = Date.now();

  const fd = new FormData(form);
  if (extra.claims) fd.set('claims', extra.claims);
  if (extra.prior) fd.set('prior', JSON.stringify(extra.prior));
  let runId;
  try {
    const res = await fetch(BASE + 'api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    runId = data.run_id;
  } catch (err) {
    fail(err.message);
    return;
  }

  const es = new EventSource(BASE + 'api/stream/' + runId);
  let finished = false;

  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'progress') {
      log(msg.message, true);
      const m = msg.data?.member;
      if (m) chipsEl.querySelector(`[data-m="${m}"]`)?.classList.add('running');
    } else if (msg.type === 'result') {
      finished = true;
      es.close();
      const r = msg.result;
      // Attach each member's published bias line for display.
      fetch(BASE + 'api/council').then(x => x.json()).then(info => {
        const byName = Object.fromEntries(info.members.map(x => [x.name, x.bias]));
        r.members.forEach(m => { m.bias_text = byName[m.member] || ''; });
        finish(r);
      }).catch(() => finish(r));
    } else if (msg.type === 'error') {
      finished = true;
      es.close();
      fail(msg.message);
    } else if (msg.type === 'done' && !finished) {
      es.close();
      fail('The run ended without producing a result.');
    }
  };

  es.onerror = () => {
    if (finished) return;
    es.close();
    fail('Lost the connection to the council.');
  };
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  startRun();
});

function finish(r) {
  (r.members || []).forEach(m => {
    const chip = chipsEl.querySelector(`[data-m="${m.member}"]`);
    if (chip) { chip.classList.remove('running'); chip.classList.add(m.scores && Object.keys(m.scores).length ? 'done' : 'failed'); }
  });
  log('Done.');
  go.disabled = false;
  go.textContent = 'Convene the council';
  lastResult = r;
  render(r);

  const btn = document.getElementById('rescan');
  if (btn) {
    btn.addEventListener('click', () => {
      const text = (document.getElementById('claims')?.value || '').trim();
      if (!text) {
        document.getElementById('claims')?.focus();
        return;
      }
      startRun({ claims: text, prior: benchmark(lastResult) });
    });
  }
}

function fail(message) {
  go.disabled = false;
  go.textContent = 'Convene the council';
  results.innerHTML = `<div class="card"><div class="err">${esc(message)}</div></div>`;
}

resetChips();
