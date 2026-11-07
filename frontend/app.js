/* PocketDiffusion frontend — vanilla JS, talks to the FastAPI backend in backend/main.py */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (ms) => (ms == null || !isFinite(ms) ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const fmtMb = (mb) => (mb == null || !isFinite(mb) ? "–" : mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${Math.round(mb)} MB`);
const num = (x, d = 1) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d));
const media = (p) => (p ? `/media/${p}` : "");
const FAMILY = { sd_turbo: "SD-Turbo", sd15: "SD1.5" };
const SCHED = { euler: "ADD", lcm: "LCM", dpm: "DPM++", pndm: "PNDM" };
const COLORS = { red: "#ef4444", orange: "#f97316", yellow: "#facc15", green: "#22c55e", blue: "#3b82f6", purple: "#a855f7",
  pink: "#ec4899", brown: "#92400e", black: "#111", white: "#fff", gray: "#9ca3af", cyan: "#06b6d4" };
const DEVICE_ICONS = {
  this_gpu: '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="10" rx="2"/><path d="M7 17v3M12 17v3M17 17v3M7 4v3M12 4v3M17 4v3"/><rect x="8" y="10" width="8" height="4" rx="1"/></svg>',
  flagship_phone: '<svg viewBox="0 0 24 24"><rect x="6" y="2" width="12" height="20" rx="3"/><path d="M10 5h4"/><path d="M9 12l2 2 4-4"/></svg>',
  midrange_phone: '<svg viewBox="0 0 24 24"><rect x="6" y="2" width="12" height="20" rx="3"/><path d="M10 5h4"/><circle cx="12" cy="12" r="2.5"/></svg>',
  lowend_phone: '<svg viewBox="0 0 24 24"><rect x="6" y="3" width="12" height="18" rx="2"/><path d="M10 6h4M9 14h6"/></svg>',
  this_cpu: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
};
const SHORT = { this_gpu: "This GPU", flagship_phone: "Flagship", midrange_phone: "Mid-range", lowend_phone: "Low-end", this_cpu: "CPU only" };
const TIMING_COLORS = { "analyse+plan": "#a78bfa", "gen.text_encode": "#f472b6", "gen.denoise": "#8b5cf6", "gen.decode": "#06b6d4",
  screen: "#22d3ee", rerank: "#34d399", semantic_rerank: "#10b981", postprocess: "#fbbf24", safety: "#fb7185" };

const state = { sys: null, device: "this_gpu", mode: "quality", busy: false, result: null, view: "final",
  gallery: { offset: 0, filter: "", q: "", total: 0 }, es: null, liveFrames: 0 };

/* ------------------------------------------------------------------ utils */
async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: { "Content-Type": "application/json" } };
  if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const d = data && data.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => `${(x.loc || []).slice(-1)[0]}: ${x.msg}`).join("; ") : `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}
function toast(msg, kind = "info", ms = 4500) {
  const t = document.createElement("div");
  t.className = `toast ${kind === "err" ? "err" : kind === "warn" ? "warn" : ""}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; setTimeout(() => t.remove(), 320); }, ms);
}
const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

/* ------------------------------------------------------------------ tabs + theme */
function showTab(name) {
  if (!$(`#view-${name}`)) name = "studio";
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "gallery") loadGallery(true);
  if (name === "lab") loadLab();
}
$$("[data-tab]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); showTab(b.dataset.tab); }));
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
function applyTheme(t) { document.documentElement.dataset.theme = t; store.set("pd-theme", t); }
applyTheme(store.get("pd-theme", "dark"));
$("#themeBtn").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* ------------------------------------------------------------------ system status */
async function refreshSystem(first = false) {
  const pill = $("#statusPill");
  try {
    const s = await api("/api/system");
    state.sys = s;
    const gpu = s.gpu ? s.gpu.name.replace("NVIDIA GeForce ", "") : (s.cloud ? "Cloud CPU (no GPU, ~10-30 s/image)" : "CPU");
    const q = s.queue ? ` · queue ${s.queue}` : "";
    pill.className = "status-pill " + (s.ready ? "ok" : "");
    $("#statusText").textContent = `${gpu} · ${s.ready ? "ready" : "warming up"}${q}`;
    if (s.gpu_link) { pill.title = `Backup copy on a free CPU server. Fast GPU version: ${s.gpu_link}`; }
    if (first) initFromSystem(s);
  } catch {
    pill.className = "status-pill bad";
    $("#statusText").textContent = "backend offline";
  }
}
function initFromSystem(s) {
  if (s.default_device && s.default_device !== state.device) {   // no GPU here: plan for the CPU
    state.device = s.default_device;
    if (state.device === "this_cpu" && msFromSlider() < 3000) setLatency(20000);
  }
  const dev = $("#devices");
  dev.innerHTML = s.device_profiles.map((p) => `
    <button class="device ${p.key === state.device ? "on" : ""}" data-dev="${p.key}" title="${esc(p.label)} — ${esc(p.notes)}">
      ${DEVICE_ICONS[p.key] || ""}${esc(SHORT[p.key] || p.label)}<small>${fmtMb(p.memory_budget_mb)}</small></button>`).join("");
  $$(".device", dev).forEach((b) => b.addEventListener("click", () => {
    state.device = b.dataset.dev;
    $$(".device", dev).forEach((x) => x.classList.toggle("on", x === b));
    const p = s.device_profiles.find((x) => x.key === state.device);
    if (p) { $("#memory").value = Math.min(8000, Math.max(500, p.memory_budget_mb)); updateSliderLabels(); }
    if (state.device === "this_cpu" && msFromSlider() < 3000) { setLatency(8000); }
    refreshPlan();
  }));
  $("#preset").innerHTML = `<option value="planner">Adaptive planner</option>` +
    s.presets.map((p) => `<option value="${esc(p.key)}">${esc(p.key)} — ${esc(p.description)}</option>`).join("");
  if (!$("#prompt").value) { $("#prompt").value = s.examples[0]; }
  refreshPlan();
}

/* ------------------------------------------------------------------ controls */
const LAT_MIN = 100, LAT_MAX = 20000;
const msFromSlider = () => Math.round(LAT_MIN * Math.pow(LAT_MAX / LAT_MIN, $("#latency").value / 1000));
const setLatency = (ms) => { $("#latency").value = Math.round(1000 * Math.log(ms / LAT_MIN) / Math.log(LAT_MAX / LAT_MIN)); updateSliderLabels(); };
function updateSliderLabels() {
  $("#latencyOut").textContent = fmtMs(msFromSlider());
  $("#memoryOut").textContent = fmtMb(+$("#memory").value);
}
setLatency(1500);
$("#latency").addEventListener("input", () => { updateSliderLabels(); refreshPlan(); });
$("#memory").addEventListener("input", () => { updateSliderLabels(); refreshPlan(); });
$$("#modes button").forEach((b) => b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  $$("#modes button").forEach((x) => x.classList.toggle("on", x === b));
  refreshPlan();
}));
["#size", "#preset"].forEach((s) => $(s).addEventListener("change", () => refreshPlan()));
$("#prompt").addEventListener("input", () => refreshPlan());
$("#diceBtn").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 2 ** 31); });
let inspireIdx = 0;
$("#inspireBtn").addEventListener("click", () => {
  const ex = state.sys?.examples || [];
  if (!ex.length) return;
  inspireIdx = (inspireIdx + 1) % ex.length;
  $("#prompt").value = ex[inspireIdx];
  refreshPlan();
});

function settings() {
  const sz = $("#size").value;
  const [w, h] = sz === "auto" ? [0, 0] : sz.split("x").map(Number);   // 0 x 0 = size named in the prompt
  const seedVal = $("#seed").value.trim();
  return {
    prompt: $("#prompt").value, device_profile: state.device, mode: state.mode, latency_ms: msFromSlider(),
    memory_mb: +$("#memory").value, width: w, height: h, negative_prompt: $("#negative").value,
    seed: seedVal === "" ? null : Math.max(0, Math.min(2 ** 31 - 1, parseInt(seedVal, 10) || 0)),
    postprocess: $("#post").value, rerank: $("#rerank").checked, verify: $("#verify").checked,
    semantic_rerank: $("#semantic").checked, preset: $("#preset").value,
  };
}

/* ------------------------------------------------------------------ live analysis + plan preview */
const refreshPlan = debounce(async () => {
  const s = settings();
  try {
    const p = await api("/api/plan", { method: "POST", body: { ...s, prompt: s.prompt || "" } });
    renderAnalysis(p.analysis, s.prompt);
    renderPlanPreview(p, s);
  } catch (e) {
    $("#planBody").innerHTML = `<span class="muted">${esc(e.message)}</span>`;
  }
}, 260);

function renderAnalysis(a, prompt) {
  const chips = $("#sceneChips");
  if (!prompt.trim()) { chips.innerHTML = `<span class="chip ghost">scene graph appears as you type</span>`; }
  else {
    const parts = [];
    a.objects.forEach((o) => parts.push(`<span class="chip">${o.color ? `<i class="sw" style="background:${COLORS[o.color] || o.color}"></i>` : ""}${esc(o.name)}${o.count > 1 ? ` <span class="n">×${o.count}</span>` : ""}</span>`));
    a.relations.forEach((r) => parts.push(`<span class="chip rel">${esc(r.subject)} → ${esc(r.relation.replace(/_/g, " "))} → ${esc(r.object)}</span>`));
    a.styles.forEach((s) => parts.push(`<span class="chip style">${esc(s)}</span>`));
    if (a.requested_size_label) parts.push(`<span class="chip style" title="Generated at ~512 px, then classical super-resolution">output ${esc(a.requested_size_label)} ${a.requested_size[0]}×${a.requested_size[1]} (upscaled)</span>`);
    (a.blocked_terms || []).forEach((b) => parts.push(`<span class="chip warn">blocked: ${esc(b)}</span>`));
    (a.warnings || []).forEach((w) => parts.push(`<span class="chip warn">${esc(w)}</span>`));
    chips.innerHTML = parts.join("") || `<span class="chip ghost">no countable objects recognised — free-form prompt</span>`;
  }
  const n = a.n_tokens || 0;
  $("#tokenCount").textContent = n;
  const arc = $("#tokenArc");
  arc.style.strokeDashoffset = 94.25 * (1 - Math.min(n / 227, 1));   // up to 3 chunks of 75 + BOS/EOS
  arc.style.stroke = a.truncated ? "var(--bad)" : (a.n_chunks || 1) > 1 ? "var(--warn)" : "var(--a2)";
  $("#tokenRing").title = `${n} tokens` + ((a.n_chunks || 1) > 1 ? ` · long-prompt mode: ${a.n_chunks} chunks` : "") +
    (a.truncated ? " · over 225 tokens, the rest is ignored" : " · CLIP reads 77 at a time; up to 3 chunks (225 tokens)");
  $("#complexityBar").style.width = `${Math.round((a.complexity || 0) * 100)}%`;
  $("#complexityLabel").textContent = prompt.trim() ? `${a.complexity_label} ${num(a.complexity, 2)}` : "–";
}

function planLabel(c) {
  const fam = FAMILY[c.family] || c.family;
  return `${fam}${c.lcm_lora ? " + LCM" : ""} · ${c.steps} step${c.steps > 1 ? "s" : ""} · ${c.width}×${c.height}`;
}
function renderPlanPreview(p, s) {
  const c = p.config;
  const lat = p.predicted_device_latency_ms ?? p.predicted_latency_ms;
  const preset = s.preset && s.preset !== "planner";
  $("#planBody").innerHTML = `
    ${preset ? `<div class="muted" style="font-size:12px">Manual model selected — the planner is bypassed. Its suggestion would be:</div>` : ""}
    <div class="big">${esc(planLabel(c))}</div>
    <div class="muted" style="font-size:12px">${p.n_candidates} candidate${p.n_candidates > 1 ? "s" : ""} · ${c.vae === "taesd" ? "TAESD" : "VAE"} · ${c.quant === "none" ? c.dtype.toUpperCase() : c.quant.toUpperCase()} · text-enc ${c.text_encoder === "t6" ? "6-layer" : "full"}${p.upscale_to ? ` · IBP→${p.upscale_to[0]}px` : ""}</div>
    <div class="kv">
      <div><small>${p.predicted_device_latency_ms != null ? "est. device" : "est. latency"}</small><b>${fmtMs(lat)}</b></div>
      <div><small>memory</small><b>${fmtMb(p.predicted_memory_mb)}</b></div>
      <div><small>CLIP≈</small><b>${num(p.predicted_clip, 1)}</b></div>
    </div>
    ${p.fallback ? `<div class="fb">⚠ Budget infeasible — resource-aware fallback plan</div>` : ""}`;
}

/* ------------------------------------------------------------------ generation */
const TRACK = ["analysis", "plan", "denoise", "screen", "postprocess", "verify", "saving"];
function trackSet(active) {
  const idx = TRACK.indexOf(active);
  $$("#track li").forEach((li, i) => {
    li.classList.toggle("active", i === idx);
    li.classList.toggle("done", idx > -1 ? i < idx : active === "done");
    li.classList.remove("skip");
  });
  if (active === "done") $$("#track li").forEach((li) => { li.classList.remove("active"); li.classList.add("done"); });
}
function setBusy(b) {
  state.busy = b;
  const go = $("#goBtn");
  go.disabled = b;
  go.classList.toggle("busy", b);
  $(".go-label", go).textContent = b ? "Generating…" : "Generate";
}
function resetRun() {
  $("#stageEmpty").classList.add("hidden");
  $("#stageFinal").classList.add("hidden");
  $("#stageLive").classList.remove("hidden");
  $("#liveImg").removeAttribute("src");
  $("#ringArc").style.strokeDashoffset = 326.7;
  $("#stepTxt").textContent = "0/0";
  $("#stageTxt").textContent = "queued";
  $("#frames").innerHTML = "";
  state.liveFrames = 0;
  trackSet(null);
}

async function generate(overrides = {}) {
  if (state.busy) return;
  const s = { ...settings(), ...overrides };
  if (!s.prompt.trim()) { toast("Please type a prompt first.", "warn"); $("#prompt").focus(); return; }
  if (s.preset === "planner") s.preset = null;
  setBusy(true);
  resetRun();
  showTab("studio");
  try {
    const { job_id } = await api("/api/generate", { method: "POST", body: s });
    listen(job_id);
  } catch (e) {
    toast(e.message, "err");
    setBusy(false);
    $("#stageLive").classList.add("hidden");
    $(state.result ? "#stageFinal" : "#stageEmpty").classList.remove("hidden");
  }
}

function listen(id) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/jobs/${id}/events`);
  state.es = es;
  let finished = false;
  const on = (name, fn) => es.addEventListener(name, (e) => { try { fn(JSON.parse(e.data || "{}")); } catch (err) { console.error(err); } });
  on("queued", (d) => { $("#stageTxt").textContent = d.busy ? `queue #${d.position}` : "queued"; });
  on("start", () => { $("#stageTxt").textContent = "analysing"; trackSet("analysis"); });
  on("analysis", () => trackSet("plan"));
  on("plan", (p) => { $("#reasons").innerHTML = p.reasons.map((r) => `<li>${esc(r)}</li>`).join(""); });
  on("stage", (d) => {
    const map = { denoise: "denoise", screen: "screen", postprocess: "postprocess", verify: "verify", saving: "saving" };
    if (map[d.name]) { trackSet(map[d.name]); $("#stageTxt").textContent = d.name === "postprocess" ? "polishing" : d.name; }
  });
  on("step", (d) => {
    $("#stepTxt").textContent = `${d.i}/${d.total}`;
    $("#ringArc").style.strokeDashoffset = 326.7 * (1 - d.i / d.total);
    if (d.preview) {
      $("#liveImg").src = d.preview;
      $("#liveImg").style.filter = `blur(${Math.max(0, 6 - (6 * d.i) / d.total)}px) saturate(1.05)`;
      const fig = document.createElement("figure");
      fig.innerHTML = `<img src="${d.preview}" alt=""><figcaption>step ${d.i}/${d.total}</figcaption>`;
      $("#frames").appendChild(fig);
    }
  });
  on("result", (g) => { finished = true; trackSet("done"); renderResult(g); });
  on("error", (d) => { if (d && d.message) { finished = true; toast(d.message, "err", 7000); } });
  on("end", (d) => {
    es.close(); state.es = null; setBusy(false);
    if (d.status === "error" || !finished) {
      $("#stageLive").classList.add("hidden");
      $(state.result ? "#stageFinal" : "#stageEmpty").classList.remove("hidden");
      trackSet(null);
    }
    refreshSystem();
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED || finished) return;
    es.close(); state.es = null; setBusy(false);
    toast("Lost connection to the server — the job may still finish; check the Gallery.", "warn");
  };
}

/* ------------------------------------------------------------------ result rendering */
function sizeLabel(w, h) {
  const m = Math.max(w, h);
  return m >= 7680 ? "8K" : m >= 3840 ? "4K" : m >= 2560 ? "2K" : m >= 1920 ? "Full HD" : `${m}px`;
}
function renderResult(g) {
  state.result = g;
  $("#stageLive").classList.add("hidden");
  $("#stageFinal").classList.remove("hidden");
  const full = g.full_path || g.image_path;   // 2K/4K/8K: full-resolution JPEG
  const fname = `pocketdiffusion-${g.id}-${g.width}x${g.height}.${full.split(".").pop()}`;
  $("#downloadBtn").href = media(full);
  $("#downloadBtn").setAttribute("download", fname);
  const hd = $("#hdDownload");                 // big outputs: make the full-resolution file obvious
  if (g.full_path) {
    hd.href = media(g.full_path);
    hd.setAttribute("download", fname);
    hd.textContent = `⤓ ${sizeLabel(g.width, g.height)} · ${g.width}×${g.height}`;
    hd.classList.remove("hidden");
  } else {
    hd.classList.add("hidden");
  }
  setView("final");

  // telemetry
  $("#latBig").textContent = fmtMs(g.latency_ms);
  const base = state.sys?.baseline_ms;
  $("#speedBadge").textContent = base ? `×${Math.max(1, Math.round(base / g.latency_ms))} faster` : "×– faster";
  $("#speedBadge").title = base ? `vs measured SD1.5 FP32 50-step baseline (${fmtMs(base)})` : "baseline not benchmarked yet";
  renderWaterfall(g.timings || {});
  renderGauge(g);
  const m = g.metrics || {};
  const d = (a, b, lowerBetter) => (a == null || b == null ? "" : `<span class="${(b - a) * (lowerBetter ? -1 : 1) >= 0 ? "up" : "down"}">${b >= a ? "▲" : "▼"}${num(Math.abs(b - a), lowerBetter ? 2 : 1)}</span>`);
  $("#miniStats").innerHTML = `
    <div><small>CLIP (B/32)</small><b>${num(g.clip, 1)}</b></div>
    <div><small>steps</small><b>${g.steps_done}/${g.steps}</b></div>
    <div><small>PocketIQA</small><b>${num(g.iqa_after, 1)}</b> ${d(g.iqa_before, g.iqa_after)}</div>
    <div><small>NIQE ↓</small><b>${num(g.niqe_after, 2)}</b> ${d(g.niqe_before, g.niqe_after, true)}</div>
    <div><small>model</small><b>${esc(FAMILY[g.family] || g.family)}</b></div>
    <div><small>N · size</small><b>${g.n_candidates} · ${g.width}px</b></div>`;
  renderChecklist(m.verification);
  $("#reasons").innerHTML = (g.plan?.reasons || ["Manual preset — planner bypassed."]).map((r) => `<li>${esc(r)}</li>`).join("");
  const ev = (g.events || []);
  $("#events").innerHTML = ev.length ? ev.map((e) => `<li>${esc(e)}</li>`).join("") : `<li class="muted">no warnings</li>`;

  // strips
  $("#frames").innerHTML = (g.frames || []).map((f) => `<figure><img src="${media(f.image_path)}" alt="" data-zoom="${media(f.image_path)}"><figcaption>step ${f.step}</figcaption></figure>`).join("") || `<span class="muted">—</span>`;
  $("#cands").innerHTML = (g.candidates || []).map((c) => `
    <figure class="${c.chosen ? "chosen" : ""}"><img src="${media(c.image_path)}" alt="" data-zoom="${media(c.image_path)}">
      <figcaption>${c.chosen ? "✔ chosen · " : ""}seed ${c.seed}${c.clip != null ? `<br>CLIP ${num(c.clip, 1)} · IQA ${num(c.iqa, 0)}` : ""}</figcaption></figure>`).join("") || `<span class="muted">—</span>`;
  const st = (g.stages || []).filter((s) => s.image_path);
  $("#stages").innerHTML = `<figure><img src="${media(g.raw_path)}" alt="" data-zoom="${media(g.raw_path)}"><figcaption>raw output</figcaption></figure>` +
    st.map((s) => `<figure><img src="${media(s.image_path)}" alt="" data-zoom="${media(s.image_path)}"><figcaption>${esc(s.name)} · ${Math.round(s.ms)} ms</figcaption></figure>`).join("") +
    (st.length ? "" : `<span class="muted" style="align-self:center">image was clean — no correction needed</span>`);
}

function renderWaterfall(t) {
  const keys = Object.keys(t).filter((k) => !["wall", "verify"].includes(k) && t[k] > 0.05);
  const total = keys.reduce((a, k) => a + t[k], 0) || 1;
  $("#waterfall").innerHTML = keys.map((k) => `<i style="width:${(100 * t[k]) / total}%;background:${TIMING_COLORS[k] || "#64748b"}" title="${esc(k)}: ${fmtMs(t[k])}"></i>`).join("");
  $("#wfLegend").innerHTML = keys.map((k) => `<span style="--c:${TIMING_COLORS[k] || "#64748b"}">${esc(k.replace("gen.", ""))} ${fmtMs(t[k])}</span>`).join("");
}
function renderGauge(g) {
  const budget = g.memory_budget || state.sys?.device_profiles.find((p) => p.key === g.device_profile)?.memory_budget_mb || 8000;
  const peak = g.peak_mem_mb || 0;
  const frac = Math.min(peak / budget, 1);
  const R = 46, C = 2 * Math.PI * R * 0.75;
  const col = frac > 0.9 ? "var(--bad)" : frac > 0.7 ? "var(--warn)" : "var(--ok)";
  $("#memGauge").innerHTML = `<svg viewBox="0 0 120 120">
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="var(--line2)" stroke-width="10" stroke-linecap="round" stroke-dasharray="${C} 999" transform="rotate(135 60 60)"/>
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="${col}" stroke-width="10" stroke-linecap="round" stroke-dasharray="${C * frac} 999" transform="rotate(135 60 60)"/>
    <text x="60" y="58" text-anchor="middle" style="font:600 16px var(--mono);fill:var(--text)">${peak ? fmtMb(peak) : "CPU"}</text>
    <text x="60" y="76" text-anchor="middle" style="font:10px var(--mono);fill:var(--muted)">of ${fmtMb(budget)}</text>
    <text x="60" y="108" text-anchor="middle" style="font:10px var(--mono);fill:var(--muted)">GPU peak · all models</text></svg>`;
}
function renderChecklist(v) {
  const el = $("#checklist");
  if (!v) { el.innerHTML = `<span class="muted">Enable "Semantic check" to verify objects, counts, colours and relations.</span>`; return; }
  const rows = [];
  rows.push(`<div class="f1">object F1 ${num(v.f1, 2)} · P ${num(v.precision, 2)} · R ${num(v.recall, 2)}</div>`);
  const want = new Set();
  (v.count_checks || []).forEach((c) => rows.push(`<div class="row ${c.ok ? "ok" : "no"}"><i>${c.ok ? "✓" : "✗"}</i>${esc(c.object)}: want ${c.expected}, found ${c.found}</div>`));
  (v.color_checks || []).forEach((c) => { want.add(c.object); rows.push(`<div class="row ${c.ok ? "ok" : "no"}"><i>${c.ok ? "✓" : "✗"}</i>${esc(c.object)} is ${esc(c.expected)}${c.found ? ` (measured ${esc(c.found)})` : ""}</div>`); });
  (v.relation_checks || []).forEach((c) => rows.push(`<div class="row ${c.ok ? "ok" : "no"}"><i>${c.ok ? "✓" : "✗"}</i>${esc(c.subject)} ${esc(c.relation.replace(/_/g, " "))} ${esc(c.object)}</div>`));
  if (v.fn) rows.push(`<div class="row no"><i>✗</i>${v.fn} requested object(s) not found</div>`);
  if (v.fp) rows.push(`<div class="row no"><i>!</i>${v.fp} hallucinated distractor class(es)</div>`);
  rows.push(`<div class="row ${v.all_satisfied ? "ok" : "no"}"><i>${v.all_satisfied ? "✓" : "✗"}</i><b>${v.all_satisfied ? "all constraints satisfied" : "some constraints violated"}</b></div>`);
  el.innerHTML = rows.join("");
}

/* view modes: final / raw / split / heat / boxes */
function setView(v) {
  const g = state.result;
  if (!g) return;
  state.view = v;
  $$("#viewToggles button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  const A = $("#imgA"), B = $("#imgB");
  A.style.clipPath = "none";
  B.classList.add("hidden");
  A.classList.remove("hidden");
  $("#splitRange").classList.add("hidden");
  $("#splitBar").style.display = "none";
  $("#boxes").innerHTML = "";
  if (v === "final" || v === "boxes") A.src = media(g.image_path);
  if (v === "raw") A.src = media(g.raw_path);
  if (v === "heat") A.src = media(g.heatmap_path);
  if (v === "split") {
    A.src = media(g.image_path); B.src = media(g.raw_path); B.classList.remove("hidden");
    $("#splitRange").classList.remove("hidden"); $("#splitBar").style.display = "block"; applySplit();
  }
  if (v === "boxes") {
    if (A.complete) drawBoxes(); else A.onload = () => { A.onload = null; drawBoxes(); };
  }
}
function applySplit() {
  const x = +$("#splitRange").value;
  const r = imageRect();
  const px = r.ox + (r.w * x) / 100;
  $("#imgA").style.clipPath = `inset(0 0 0 ${px}px)`;
  $("#splitBar").style.left = `${px}px`;
}
function imageRect() {
  const wrap = $("#compareWrap"), A = $("#imgA");
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const iw = A.naturalWidth || 512, ih = A.naturalHeight || 512;
  const s = Math.min(cw / iw, ch / ih);
  return { ox: (cw - iw * s) / 2, oy: (ch - ih * s) / 2, w: iw * s, h: ih * s, s };
}
function drawBoxes() {
  const g = state.result;
  const v = g?.metrics?.verification;
  const el = $("#boxes");
  if (!v || !v.detections?.length) { el.innerHTML = ""; toast(v ? "No objects detected." : "Run with Semantic check enabled to see detections.", "warn"); return; }
  const wanted = new Set((g.analysis?.objects || []).map((o) => o.name));
  const r = imageRect();
  el.innerHTML = v.detections.map((d) => {
    const [x0, y0, x1, y1] = d.box;
    return `<div class="bx ${wanted.has(d.label) ? "" : "fp"}" style="left:${r.ox + x0 * r.s}px;top:${r.oy + y0 * r.s}px;width:${(x1 - x0) * r.s}px;height:${(y1 - y0) * r.s}px"><span>${esc(d.label)} ${num(d.score, 2)}</span></div>`;
  }).join("");
}
$$("#viewToggles button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
$("#splitRange").addEventListener("input", applySplit);
window.addEventListener("resize", debounce(() => { if (state.view === "boxes") drawBoxes(); if (state.view === "split") applySplit(); }, 120));
$("#rerollBtn").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 2 ** 31); generate(); });
$("#goBtn").addEventListener("click", () => generate());
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && $("#view-studio").classList.contains("active")) { e.preventDefault(); generate(); }
  if (e.key === "Escape") closeModal();
});
document.addEventListener("click", (e) => {
  const z = e.target.closest("[data-zoom]");
  if (z) openModal(`<img src="${esc(z.dataset.zoom)}" alt="" style="width:100%;border-radius:14px;image-rendering:auto">`);
});

/* ------------------------------------------------------------------ modal */
function openModal(html) { $("#modalBody").innerHTML = html; $("#modal").classList.remove("hidden"); }
function closeModal() { $("#modal").classList.add("hidden"); $("#modalBody").innerHTML = ""; }
$("#modalClose").addEventListener("click", closeModal);
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });

/* ------------------------------------------------------------------ gallery */
async function loadGallery(reset = false) {
  const G = state.gallery;
  if (reset) { G.offset = 0; $("#gGrid").innerHTML = ""; }
  const p = new URLSearchParams({ limit: 24, offset: G.offset });
  if (G.filter === "fav") p.set("favorites", "true");
  else if (G.filter) p.set("family", G.filter);
  if (G.q) p.set("q", G.q);
  try {
    const { items, total } = await api(`/api/generations?${p}`);
    G.total = total;
    G.offset += items.length;
    $("#gCount").textContent = `${total} image${total === 1 ? "" : "s"}`;
    if (!total) { $("#gGrid").innerHTML = `<div class="empty-state">No generations yet — create one in the Studio.</div>`; }
    $("#gGrid").insertAdjacentHTML("beforeend", items.map(cardHtml).join(""));
    $("#gMore").classList.toggle("hidden", G.offset >= total);
  } catch (e) { toast(`Gallery: ${e.message}`, "err"); }
}
function cardHtml(g) {
  return `<div class="gcard" data-id="${esc(g.id)}">
    <img loading="lazy" src="${media(g.image_path)}" alt="${esc(g.prompt)}">
    <button class="fav ${g.favorite ? "on" : ""}" data-fav="${esc(g.id)}" title="Favourite">${g.favorite ? "★" : "☆"}</button>
    <div class="meta"><p>${esc(g.prompt)}</p><div class="badges">
      <span>${esc(FAMILY[g.family] || g.family)} · ${g.steps}st</span><span>${fmtMs(g.latency_ms)}</span>${g.clip != null ? `<span>CLIP ${num(g.clip, 1)}</span>` : ""}${g.object_f1 != null ? `<span>F1 ${num(g.object_f1, 2)}</span>` : ""}</div></div></div>`;
}
$("#gGrid").addEventListener("click", async (e) => {
  const fav = e.target.closest("[data-fav]");
  if (fav) {
    e.stopPropagation();
    try { const r = await api(`/api/generations/${fav.dataset.fav}/favorite`, { method: "POST" }); fav.classList.toggle("on", r.favorite); fav.textContent = r.favorite ? "★" : "☆"; }
    catch (err) { toast(err.message, "err"); }
    return;
  }
  const card = e.target.closest(".gcard");
  if (card) openDetail(card.dataset.id);
});
$$("#gFilter button").forEach((b) => b.addEventListener("click", () => {
  state.gallery.filter = b.dataset.f;
  $$("#gFilter button").forEach((x) => x.classList.toggle("on", x === b));
  loadGallery(true);
}));
$("#gSearch").addEventListener("input", debounce((e) => { state.gallery.q = e.target.value.trim(); loadGallery(true); }, 300));
$("#gMore").addEventListener("click", () => loadGallery(false));

async function openDetail(id) {
  try {
    const g = await api(`/api/generations/${id}`);
    const c = g.config || {};
    const kv = [["model", `${FAMILY[g.family] || g.family} · ${SCHED[g.scheduler] || g.scheduler} ${g.steps_done}/${g.steps}`],
      ["size", `${g.width}×${g.height}`], ["latency", fmtMs(g.latency_ms)], ["peak memory", fmtMb(g.peak_mem_mb)],
      ["weights", `${fmtMb(g.weights_mb)} · ${c.quant === "none" ? (c.dtype || "").toUpperCase() : (c.quant || "").toUpperCase()}`],
      ["decoder", g.decoder === "taesd" ? "TAESD" : "VAE"], ["candidates", g.n_candidates], ["seed", g.seed],
      ["CLIP (B/32)", num(g.clip, 1)], ["PocketIQA", `${num(g.iqa_before, 1)} → ${num(g.iqa_after, 1)}`],
      ["NIQE", `${num(g.niqe_before, 2)} → ${num(g.niqe_after, 2)}`], ["object F1", num(g.object_f1, 2)],
      ["device", SHORT[g.device_profile] || g.device_profile], ["mode", g.mode], ["created", new Date(g.created_at * 1000).toLocaleString()]];
    openModal(`<div class="detail">
      <div><img class="main" src="${media(g.image_path)}" alt=""></div>
      <div><h3>${esc(g.prompt)}</h3>
        <div class="kvs">${kv.map(([k, v]) => `<div><small>${esc(k)}</small>${esc(v)}</div>`).join("")}</div>
        <div class="strip cands" style="margin-top:12px">${(g.candidates || []).map((x) => `<figure class="${x.chosen ? "chosen" : ""}"><img src="${media(x.image_path)}" alt=""><figcaption>${x.chosen ? "✔ " : ""}${x.clip != null ? `CLIP ${num(x.clip, 1)}` : `seed ${x.seed}`}</figcaption></figure>`).join("")}</div>
        <div class="actions">
          <button class="btn primary" id="dRemix">Remix in Studio</button>
          <button class="btn" id="dFav">${g.favorite ? "★ Unfavourite" : "☆ Favourite"}</button>
          <a class="btn" href="${media(g.full_path || g.image_path)}" download="pocketdiffusion-${esc(g.id)}-${g.width}x${g.height}.${(g.full_path || g.image_path).split(".").pop()}">Download ${g.width}×${g.height}</a>
          <button class="btn danger" id="dDel">Delete</button>
        </div></div></div>`);
    $("#dRemix").onclick = () => { remix(g); closeModal(); };
    $("#dFav").onclick = async () => { const r = await api(`/api/generations/${g.id}/favorite`, { method: "POST" }); $("#dFav").textContent = r.favorite ? "★ Unfavourite" : "☆ Favourite"; loadGallery(true); };
    $("#dDel").onclick = async () => {
      if (!confirm("Delete this generation and its images?")) return;
      try { await api(`/api/generations/${g.id}`, { method: "DELETE" }); closeModal(); toast("Deleted."); loadGallery(true); } catch (err) { toast(err.message, "err"); }
    };
  } catch (e) { toast(e.message, "err"); }
}
function remix(g) {
  $("#prompt").value = g.prompt;
  $("#seed").value = g.seed ?? "";
  $("#negative").value = g.negative_prompt || "";
  if (g.mode) { state.mode = g.mode; $$("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === g.mode)); }
  if (g.device_profile) { state.device = g.device_profile; $$(".device").forEach((b) => b.classList.toggle("on", b.dataset.dev === g.device_profile)); }
  if (g.latency_budget) setLatency(g.latency_budget);
  if (g.memory_budget) { $("#memory").value = g.memory_budget; updateSliderLabels(); }
  showTab("studio");
  refreshPlan();
  state.result = g; renderResult(g); $("#stageEmpty").classList.add("hidden");
}

/* ------------------------------------------------------------------ SVG charts */
const LABELS = { baseline: "Baseline FP32·50", sd15_fp16: "+FP16", sd15_fp16_dpm20: "+DPM++20", sd15_lcm4: "+LCM 4", sd15_lcm4_taesd: "+TAESD",
  sd15_lcm4_taesd_int8: "+INT8", sd15_lcm4_taesd_int4: "+INT4", sd15_lcm4_taesd_int8_t6: "+6L text-enc", turbo1_taesd: "SD-Turbo 1",
  turbo1_taesd_int8: "Turbo INT8", turbo1_taesd_int4: "Turbo INT4", pocketdiff: "PocketDiffusion", pocketdiff_fast: "Pocket fast",
  pocketdiff_semantic: "Pocket+semantic" };
const lab = (k) => LABELS[k] || k;
function hbar(el, items, { log = false, unit = "ms", fmt = (v) => fmtMs(v) } = {}) {
  if (!items.length) { el.innerHTML = `<p class="muted">no data yet — run <code>python -m eval.run_all</code></p>`; return; }
  const W = 560, rowH = 24, L = 150, R = 80, H = items.length * rowH + 10;
  const max = Math.max(...items.map((i) => (i.value || 0) + (i.value2 || 0)));
  const min = log ? Math.min(...items.map((i) => i.value)) * 0.7 : 0;
  const sx = (v) => log ? ((Math.log(Math.max(v, min)) - Math.log(min)) / (Math.log(max * 1.05) - Math.log(min))) * (W - L - R) : (v / (max * 1.05)) * (W - L - R);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${items.map((it, i) => {
    const y = i * rowH + 4, w1 = Math.max(2, sx(it.value)), w2 = it.value2 ? Math.max(0, sx(it.value + it.value2) - w1) : 0;
    return `<text x="${L - 8}" y="${y + 14}" text-anchor="end" class="lbl">${esc(it.label)}</text>
      <rect x="${L}" y="${y + 3}" width="${w1}" height="15" rx="4" fill="${it.color || "url(#hg)"}"/>
      ${w2 ? `<rect x="${L + w1}" y="${y + 3}" width="${w2}" height="15" rx="4" fill="var(--a3)" opacity=".55"/>` : ""}
      <text x="${L + w1 + w2 + 6}" y="${y + 14}">${esc(it.text || fmt(it.value))}</text>`;
  }).join("")}<defs><linearGradient id="hg"><stop offset="0" stop-color="var(--a1)"/><stop offset="1" stop-color="var(--a2)"/></linearGradient></defs></svg>`;
}
function scatter(el, pts) {
  if (!pts.length) { el.innerHTML = `<p class="muted">no data yet — run <code>python -m eval.run_all</code></p>`; return; }
  const W = 980, H = 340, L = 50, R = 20, T = 14, B = 40;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x0 = Math.log10(Math.min(...xs) * 0.6), x1 = Math.log10(Math.max(...xs) * 1.6);
  const y0 = Math.floor(Math.min(...ys) - 0.6), y1 = Math.ceil(Math.max(...ys) + 0.6);
  const sx = (v) => L + ((Math.log10(v) - x0) / (x1 - x0)) * (W - L - R), sy = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000].filter((t) => Math.log10(t) >= x0 && Math.log10(t) <= x1);
  const yt = []; for (let v = y0; v <= y1; v += 1) yt.push(v);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    ${ticks.map((t) => `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${T}" y2="${H - B}"/><text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${fmtMs(t)}</text>`).join("")}
    ${yt.map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end">${v}</text>`).join("")}
    <text x="${(W + L) / 2}" y="${H - 4}" text-anchor="middle">generation latency (log)</text>
    <text x="14" y="${T + (H - T - B) / 2}" transform="rotate(-90 14 ${T + (H - T - B) / 2})" text-anchor="middle">CLIP-L</text>
    ${pts.map((p) => `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="${Math.max(5, Math.sqrt(p.r || 1000) / 4)}" fill="${p.hl ? "var(--a3)" : p.base ? "#64748b" : "var(--a1)"}" fill-opacity=".55" stroke="${p.hl ? "var(--a3)" : "var(--a2)"}" stroke-width="1.5"><title>${esc(p.label)}: ${fmtMs(p.x)}, CLIP ${num(p.y, 2)}, ${fmtMb(p.r)}</title></circle>
      <text x="${sx(p.x) + 9}" y="${sy(p.y) - 8}" class="lbl">${esc(p.label)}</text>`).join("")}
  </svg>`;
}
function grouped(el, systems, metrics) {
  if (!systems.length) { el.innerHTML = `<p class="muted">no data yet — run <code>python -m eval.eval_compositional</code></p>`; return; }
  const W = 560, H = 250, L = 34, B = 46, T = 10, gw = (W - L) / metrics.length;
  const palette = ["#64748b", "#a78bfa", "#8b5cf6", "#06b6d4", "#f472b6", "#fbbf24"];
  const bw = Math.min(16, (gw - 14) / systems.length);
  const sy = (v) => T + (1 - v) * (H - T - B);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H + 24 + Math.ceil(systems.length / 3) * 16}">
    ${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line class="grid" x1="${L}" x2="${W}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${v}</text>`).join("")}
    ${metrics.map((m, mi) => systems.map((s, si) => {
      const v = s.v[m.key] ?? 0, x = L + mi * gw + 7 + si * bw;
      return `<rect x="${x}" y="${sy(v)}" width="${bw - 2}" height="${H - B - sy(v)}" rx="3" fill="${palette[si % palette.length]}"><title>${esc(s.name)} · ${m.label}: ${num(v, 3)}</title></rect>`;
    }).join("") + `<text x="${L + mi * gw + gw / 2}" y="${H - B + 16}" text-anchor="middle" class="lbl">${m.label}</text>`).join("")}
    ${systems.map((s, si) => `<rect x="${L + (si % 3) * 180}" y="${H + 4 + Math.floor(si / 3) * 16}" width="10" height="10" rx="2" fill="${palette[si % palette.length]}"/><text x="${L + (si % 3) * 180 + 14}" y="${H + 13 + Math.floor(si / 3) * 16}">${esc(s.name)}</text>`).join("")}
  </svg>`;
}
function lines(el, series, { xlabel = "", ylabel = "", logx = false } = {}) {
  const all = series.flatMap((s) => s.pts);
  if (!all.length) { el.innerHTML = `<p class="muted">no data yet</p>`; return; }
  const W = 560, H = 240, L = 44, R = 90, T = 12, B = 36;
  const fx = (v) => (logx ? Math.log2(v) : v);
  const xs = all.map((p) => fx(p[0])), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1, y0 = Math.floor(Math.min(...ys) - 0.5), y1 = Math.ceil(Math.max(...ys) + 0.5);
  const sx = (v) => L + ((fx(v) - x0) / ((x1 - x0) || 1)) * (W - L - R), sy = (v) => T + (1 - (v - y0) / ((y1 - y0) || 1)) * (H - T - B);
  const pal = ["var(--a1)", "var(--a2)", "var(--a3)", "var(--warn)"];
  const xt = [...new Set(all.map((p) => p[0]))].sort((a, b) => a - b);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    ${[y0, (y0 + y1) / 2, y1].map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${num(v, 0)}</text>`).join("")}
    ${xt.map((v) => `<text x="${sx(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`).join("")}
    <text x="${(W + L - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xlabel)}</text>
    <text x="12" y="${(H - B) / 2}" transform="rotate(-90 12 ${(H - B) / 2})" text-anchor="middle">${esc(ylabel)}</text>
    ${series.map((s, i) => `<polyline fill="none" stroke="${pal[i % 4]}" stroke-width="2.5" points="${s.pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")}"/>
      ${s.pts.map((p) => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="${pal[i % 4]}"><title>${esc(s.name)} ${p[0]}: ${num(p[1], 2)}</title></circle>`).join("")}
      <text x="${sx(s.pts[s.pts.length - 1][0]) + 8}" y="${sy(s.pts[s.pts.length - 1][1]) + 4}" class="lbl">${esc(s.name)}</text>`).join("")}
  </svg>`;
}

async function loadLab() {
  try {
    const [st, bm] = await Promise.all([api("/api/stats"), api("/api/benchmarks")]);
    const e = bm.efficiency || {}, q = bm.quality || {}, comp = bm.compositional || {}, deg = bm.degradation || {};
    const base = e.baseline, turbo = e.turbo1_taesd;
    const tiles = [
      ["generations", st.n || 0, "stored in SQLite"],
      ["avg latency", fmtMs(st.lat), st.best ? `best ${fmtMs(st.best)}` : ""],
      ["avg CLIP", num(st.clip, 1), "ViT-B/32 re-ranker"],
      ["time saved", st.time_saved_s != null ? `${num(st.time_saved_s, 0)} s` : "–", "vs. running the baseline"],
      ["fastest config", turbo ? fmtMs(turbo.latency_ms_mean) : "–", base && turbo ? `×${Math.round(base.latency_ms_mean / turbo.latency_ms_mean)} vs baseline` : ""],
      ["smallest model", e.sd15_lcm4_taesd_int4 ? fmtMb(e.sd15_lcm4_taesd_int4.size_total_mb) : "–", base ? `baseline ${fmtMb(base.size_total_mb)}` : ""],
    ];
    $("#statTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="card tile"><small>${esc(k)}</small><b>${esc(v)}</b><span>${esc(s)}</span></div>`).join("");

    scatter($("#chScatter"), Object.entries(q).filter(([k, v]) => !k.startsWith("_") && v.gen_latency_ms).map(([k, v]) => ({
      label: lab(k), x: v.gen_latency_ms.mean, y: v.clip_mean, r: v.peak_mem_mb, hl: k.startsWith("pocketdiff"), base: k === "baseline" })));
    const effKeys = Object.keys(e).filter((k) => !k.startsWith("_") && !k.startsWith("cpu") && e[k].latency_ms_mean);
    hbar($("#chLatency"), effKeys.map((k) => ({ label: lab(k), value: e[k].latency_ms_mean })), { log: true });
    hbar($("#chMemory"), effKeys.map((k) => ({ label: lab(k), value: e[k].size_total_mb, value2: Math.max(0, e[k].peak_mem_mb - e[k].size_total_mb), text: `${fmtMb(e[k].peak_mem_mb)} peak` })), { fmt: fmtMb });
    const sys = Object.entries(comp).filter(([k, v]) => !k.startsWith("_") && v.object_f1 != null).map(([k, v]) => ({ name: lab(k), v }));
    grouped($("#chComp"), sys, [{ key: "object_f1", label: "object F1" }, { key: "count_acc", label: "count" }, { key: "color_acc", label: "colour" },
      { key: "spatial_acc", label: "spatial" }, { key: "all_constraints_rate", label: "all ok" }]);
    const ser = [];
    const t = Object.entries(deg).filter(([k]) => k.startsWith("turbo_steps")).map(([, v]) => [v.steps, v.clip_mean]).sort((a, b) => a[0] - b[0]);
    const l = Object.entries(deg).filter(([k]) => k.startsWith("lcm_steps")).map(([, v]) => [v.steps, v.clip_mean]).sort((a, b) => a[0] - b[0]);
    if (t.length) ser.push({ name: "SD-Turbo", pts: t });
    if (l.length) ser.push({ name: "SD1.5+LCM", pts: l });
    lines($("#chDeg"), ser, { xlabel: "sampling steps", ylabel: "CLIP-L", logx: true });
    const rec = (st.recent || []).map((r, i) => [i + 1, r.latency_ms]);
    if (rec.length > 1) lines($("#chSession"), [{ name: "latency ms", pts: rec }], { xlabel: "generation #", ylabel: "ms" });
    else $("#chSession").innerHTML = `<p class="muted">Generate a few images in the Studio — this chart is read live from the database.</p>`;
  } catch (err) { toast(`Lab: ${err.message}`, "err"); }
}

/* ------------------------------------------------------------------ race */
// memory this generation added on top of what was already resident (weights of every loaded model)
const extraMb = (g) => Math.max(0, (g.peak_mem_mb || 0) - (g.metrics?.resident_mb || 0));
let raceTimers = {};
function laneTimer(lane, startMs) {
  cancelAnimationFrame(raceTimers[lane]?.raf);
  const el = lane === "optimised" ? $("#tOpt") : $("#tBase");
  const t0 = performance.now() - (startMs || 0);
  const tick = () => { el.textContent = `${((performance.now() - t0) / 1000).toFixed(2)} s`; raceTimers[lane].raf = requestAnimationFrame(tick); };
  raceTimers[lane] = { raf: requestAnimationFrame(tick) };
}
function laneStop(lane, ms) {
  cancelAnimationFrame(raceTimers[lane]?.raf);
  (lane === "optimised" ? $("#tOpt") : $("#tBase")).textContent = fmtMs(ms);
}
$("#raceBtn").addEventListener("click", async () => {
  const prompt = $("#raceprompt").value.trim();
  if (!prompt) { toast("Type a prompt for the race.", "warn"); return; }
  if (state.busy) { toast("A generation is running — wait for it to finish.", "warn"); return; }
  ["#pOpt", "#pBase"].forEach((s) => ($(s).style.width = "0"));
  ["#iOpt", "#iBase"].forEach((s) => $(s).removeAttribute("src"));
  ["#mOpt", "#mBase"].forEach((s) => ($(s).textContent = ""));
  $("#tOpt").textContent = $("#tBase").textContent = "0.00 s";
  $("#raceResult").classList.add("hidden");
  $("#raceBtn").disabled = true;
  try {
    const { job_id } = await api("/api/compare", { method: "POST", body: { prompt, seed: +$("#raceSeed").value || 7 } });
    const es = new EventSource(`/api/jobs/${job_id}/events`);
    es.addEventListener("lane", (ev) => {
      const d = JSON.parse(ev.data);
      const bar = d.lane === "optimised" ? $("#pOpt") : $("#pBase");
      if (d.state === "start") { laneTimer(d.lane, 0); $(d.lane === "optimised" ? "#mOpt" : "#mBase").textContent = d.lane === "baseline" ? "loading baseline weights…" : "planning…"; }
      if (d.total) { bar.style.width = `${(100 * d.i) / d.total}%`; $(d.lane === "optimised" ? "#mOpt" : "#mBase").textContent = `step ${d.i}/${d.total}`; }
      if (d.state === "done" && d.generation) {
        const g = d.generation;
        bar.style.width = "100%";
        laneStop(d.lane, g.latency_ms);
        $(d.lane === "optimised" ? "#iOpt" : "#iBase").src = media(g.image_path);
        $(d.lane === "optimised" ? "#mOpt" : "#mBase").textContent = `${FAMILY[g.family]} · ${g.steps} step${g.steps > 1 ? "s" : ""} · ${g.width}px · ${g.n_candidates} candidate${g.n_candidates > 1 ? "s" : ""} · weights ${fmtMb(g.weights_mb)} (model loading excluded)`;
      }
    });
    es.addEventListener("result", (ev) => {
      const r = JSON.parse(ev.data), b = r.baseline, o = r.optimised;
      $("#raceResult").innerHTML = `<div class="huge">×${num(r.speedup, 0)} faster</div>
        <table><tr><th></th><th>Baseline</th><th>PocketDiffusion</th></tr>
        <tr><td>latency</td><td>${fmtMs(b.latency_ms)}</td><td>${fmtMs(o.latency_ms)}</td></tr>
        <tr><td>UNet evaluations</td><td>${r.unet_evals[0]}</td><td>${r.unet_evals[1]}</td></tr>
        <tr><td>extra GPU memory for this run</td><td>${fmtMb(extraMb(b))}</td><td>${fmtMb(extraMb(o))}</td></tr>
        <tr><td>model weights</td><td>${fmtMb(b.weights_mb)}</td><td>${fmtMb(o.weights_mb)}</td></tr>
        <tr><td>CLIP (B/32)</td><td>${num(r.clip_baseline, 1)}</td><td>${num(r.clip_optimised, 1)}</td></tr></table>
        <p class="muted" style="font-size:12px">Live demo baseline runs in FP16 to stay responsive; the report's baseline is FP32 (13.3 s).
        This server keeps several models resident, so isolated peak-memory numbers are in the Lab tab.</p>`;
      $("#raceResult").classList.remove("hidden");
    });
    es.addEventListener("error", (ev) => { if (ev.data) { try { toast(JSON.parse(ev.data).message, "err"); } catch { /* transport error */ } } });
    es.addEventListener("end", () => { es.close(); $("#raceBtn").disabled = false; ["optimised", "baseline"].forEach((l) => cancelAnimationFrame(raceTimers[l]?.raf)); });
    es.onerror = () => { if (es.readyState !== EventSource.CLOSED) { es.close(); $("#raceBtn").disabled = false; } };
  } catch (e) { toast(e.message, "err"); $("#raceBtn").disabled = false; }
});

/* ------------------------------------------------------------------ boot */
updateSliderLabels();
refreshSystem(true);
setInterval(refreshSystem, 5000);
showTab(location.hash.slice(1) || "studio");
