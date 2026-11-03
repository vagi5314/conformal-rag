/* Conformal RAG console.
   Reads the REST API and paints four views: Check, Topics, Status, Method.
   No framework, no build step. */

const HTML_NS = "http://www.w3.org/2000/svg";
const ALPHA_CHOICES = [0.2, 0.15, 0.1, 0.05];
const MIN_SAMPLES = [
  { alpha: 0.2, n: 4 },
  { alpha: 0.15, n: 6 },
  { alpha: 0.1, n: 9 },
  { alpha: 0.05, n: 19 },
  { alpha: 0.01, n: 99 },
];
const TOPIC_BLURB = {
  general: "Everyday questions with no domain specific risk.",
  medicine: "Dosing, symptoms and treatment, where a wrong answer costs the most.",
  finance: "Definitions and ratios, where an error is usually recoverable.",
};

const state = {
  config: null,
  calibration: null,
  domains: [],
  demo: null,
  metrics: null,
  activeTopic: null,
  activeKind: null,
  chartTopic: "general",
  toleranceTopic: "general",
  doc: null,
  busy: false,
};

/* ------------------------------------------------------------- helpers */

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  });
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(HTML_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

const score = (x) => Number(x).toFixed(4);
const alpha = (x) => Number(x).toFixed(2);
const pct = (x) => `${(Number(x) * 100).toFixed(1)}%`;
const title = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function clamp01(x) {
  return Math.min(1, Math.max(0, Number(x)));
}

function useIcon(node, name) {
  node.replaceChildren(svgEl("use", { href: `#i-${name}` }));
  return node;
}

function showError(node, message) {
  node.textContent = message;
  node.hidden = false;
}

/* ----------------------------------------------------------- boot / nav */

const VIEW_IDS = ["check", "topics", "status", "method"];

function activate(id, focus) {
  VIEW_IDS.forEach((name) => {
    const view = $(`view-${name}`);
    const tab = $(`tab-${name}`);
    const on = name === id;
    view.hidden = !on;
    if (on) {
      view.classList.remove("is-entering");
      void view.offsetWidth;
      view.classList.add("is-entering");
    }
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  });
  if (focus) $(`view-${id}`).focus({ preventScroll: true });
  if (id === "status") loadStatus();
  if (id === "topics") loadDomains();
}

function bindNav() {
  const tabs = VIEW_IDS.map((id) => $(`tab-${id}`));
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.view, false));
    tab.addEventListener("keydown", (event) => {
      const i = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      if (event.key === "Home") next = tabs[0];
      if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) {
        event.preventDefault();
        next.focus();
        activate(next.dataset.view, false);
      }
    });
  });
}

/* ------------------------------------------------------------- loading */

async function loadConfig() {
  state.config = await API.get("/config");
  renderTopbarState();
  renderFooter();
  renderTolerance();
  renderFormula();
}

async function loadCalibration() {
  state.calibration = await API.get("/calibration");
  renderTolerance();
}

async function loadDomains() {
  const data = await API.get("/domains");
  state.domains = data.domains;
  renderDomains();
  renderTopicRow();
  renderChartTopicRow();
}

async function loadDemo() {
  state.demo = await API.get("/demo");
  state.activeTopic = Object.keys(state.demo.showcase)[0];
  if (!state.activeKind) {
    const first = state.demo.showcase[state.activeTopic].cases[0];
    state.activeKind = first ? first.kind : null;
  }
  renderPresets();
}

async function loadStatus() {
  try {
    state.metrics = await API.get("/metrics");
  } catch (err) {
    state.metrics = null;
  }
  if (!state.domains.length) {
    try {
      await loadDomains();
    } catch (err) {
      /* the table renders its own empty state */
    }
  }
  renderMetrics();
  renderDriftTable();
  await renderSeparation();
}

/* ------------------------------------------------------ check: presets */

function renderPresets() {
  if (!state.demo) return;
  const topics = $("preset-row");
  const kinds = $("kind-row");
  topics.replaceChildren();
  kinds.replaceChildren();

  Object.entries(state.demo.showcase).forEach(([topic, spec]) => {
    const on = topic === state.activeTopic;
    const chip = el("button", {
      type: "button",
      class: `chip button-xsm${on ? " is-active" : ""}`,
      "aria-pressed": String(on),
    }, title(topic));
    chip.addEventListener("click", () => {
      state.activeTopic = topic;
      state.activeKind = null;
      renderPresets();
    });
    topics.appendChild(chip);
  });

  state.demo.showcase[state.activeTopic].cases.forEach((item) => {
    const on = item.kind === state.activeKind;
    const chip = el("button", {
      type: "button",
      class: `chip button-xsm${on ? " is-active" : ""}`,
      "aria-pressed": String(on),
    });
    chip.appendChild(el("span", { class: `chip-dot chip-dot--${item.kind}` }));
    chip.appendChild(el("span", {}, `${item.kind} answer`));
    chip.addEventListener("click", () => {
      state.activeKind = item.kind;
      fillForm(state.activeTopic, item.kind);
      renderPresets();
    });
    kinds.appendChild(chip);
  });

  if (state.activeKind) {
    fillForm(state.activeTopic, state.activeKind);
  } else {
    loadPresetIntoForm();
  }
}

function loadPresetIntoForm() {
  const spec = state.demo.showcase[state.activeTopic];
  $("f-query").value = spec.query;
  $("f-context").value = spec.context;
  $("f-response").value = "";
  $("check-domain-note").textContent = `Auto-routes by topic. Example loaded: ${state.activeTopic}.`;
  state.toleranceTopic = state.activeTopic;
  renderTolerance();
}

function fillForm(topic, kind) {
  const spec = state.demo.showcase[topic];
  const item = spec.cases.find((c) => c.kind === kind);
  $("f-query").value = spec.query;
  $("f-context").value = spec.context;
  $("f-response").value = item ? item.response : "";
  $("check-error").hidden = true;
  $("check-domain-note").textContent = `Example loaded: ${topic}, ${kind} answer.`;
  state.toleranceTopic = topic;
  renderTolerance();
}

/* --------------------------------------------------------------- check */

function bindCheckForm() {
  $("check-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runCheck();
  });
  $("check-clear").addEventListener("click", () => {
    $("f-response").value = "";
    $("check-error").hidden = true;
    $("result-card").hidden = true;
  });
}

async function runCheck() {
  if (state.busy) return;
  const query = $("f-query").value.trim();
  const context = $("f-context").value.trim();
  const response = $("f-response").value.trim();
  const errorNode = $("check-error");

  if (!query || !context || !response) {
    showError(errorNode, "Fill in the question, the source passage and the answer before checking.");
    return;
  }
  errorNode.hidden = true;

  state.busy = true;
  const submit = $("check-submit");
  const originalLabel = submit.textContent;
  submit.disabled = true;
  submit.textContent = "Checking";
  $("result-card").hidden = false;
  $("result-skeleton").hidden = false;
  $("result-body").hidden = true;
  $("result-latency").textContent = "";

  try {
    const result = await API.post("/predict", { query, context: [context], response });
    renderResult(result);
  } catch (err) {
    $("result-card").hidden = true;
    showError(errorNode, err.message);
  } finally {
    state.busy = false;
    submit.disabled = false;
    submit.textContent = originalLabel;
    $("result-skeleton").hidden = true;
  }
}

function renderResult(result, options = {}) {
  const card = $("result-card");
  const fromDocument = options.fromDocument === true;
  card.hidden = false;
  $("result-body").hidden = false;
  $("result-band").hidden = false;
  $("fact-score").hidden = false;

  const approved = !result.abstained;
  const calibrated = result.calibrated !== false;
  const verdict = $("result-verdict");
  verdict.classList.toggle("is-approved", approved && calibrated);
  verdict.classList.toggle("is-withheld", !approved || !calibrated);
  useIcon($("result-icon"), approved && calibrated ? "check" : "withhold");
  $("result-verdict-text").textContent = calibrated
    ? (approved ? "Approved" : "Withheld")
    : "No bound";

  const threshold = clamp01(result.threshold === null || result.threshold === undefined ? 1 : result.threshold);
  const band = card.querySelector(".band");
  const topicAlpha = result.alpha !== null && result.alpha !== undefined
    ? result.alpha
    : topicTolerance(result.domain);

  if (!calibrated) {
    $("result-meaning").textContent =
      `The ${result.domain} topic has no calibration, so this score carries no guarantee. ` +
      `The threshold shown is a plain default, not a certified bound.`;
  } else if (approved) {
    $("result-meaning").textContent = fromDocument
      ? `The score cleared the ${result.domain} threshold, so the check approves this answer. ` +
        `That threshold comes from the demo calibration set, not from this document.`
      : `The score cleared the threshold, so this answer is covered by the promise for ${result.domain}. ` +
        `Among answers approved at a tolerance of ${alpha(topicAlpha)}, the expected share that are wrong stays at or below ${alpha(topicAlpha)}.`;
  } else {
    $("result-meaning").textContent =
      `The score fell short of the threshold, so nothing is claimed about this answer. It may still be right. ` +
      `The system withholds instead of approving something the bound cannot cover.`;
  }

  const approvedStart = threshold * 100;
  const width = Math.max(0, 100 - approvedStart);
  $("band-approved").style.left = `${approvedStart}%`;
  $("band-approved").style.width = `${width}%`;
  $("band-threshold").style.left = `${approvedStart}%`;
  $("band-marker").style.left = `${clamp01(result.risk_score) * 100}%`;
  band.setAttribute(
    "aria-label",
    `Score ${score(result.risk_score)} against a threshold of ${score(threshold)}.`
  );

  $("result-score").textContent = score(result.risk_score);
  $("result-threshold").textContent = score(threshold);
  $("result-domain").textContent = result.domain;
  $("result-latency").textContent = `${Math.round(result.latency_ms)} ms`;

  const gap = result.risk_score - threshold;
  const parts = [];
  parts.push(
    gap >= 0
      ? `Score sits ${score(gap)} above the ${result.domain} threshold.`
      : `Score sits ${score(Math.abs(gap))} below the ${result.domain} threshold.`
  );
  const breakdown = result.scoring_breakdown || {};
  if (breakdown.used_llm_judge) {
    parts.push("A language model judge reviewed this one because the first score landed in the uncertain band.");
  } else if (breakdown.judge_failed) {
    parts.push("The judge was unavailable, so the score rests on the entailment model alone.");
  }
  if (options.note) parts.push(options.note);
  $("result-note").textContent = parts.join(" ");

  state.toleranceTopic = result.domain;
  renderTolerance();
}

/* ------------------------------------------------- check: document mode */

const DOC_EXT = /\.(pdf|txt|md|text)$/i;
const MAX_UPLOAD_MB = 10;

function bindModeSwitch() {
  const paste = $("mode-paste");
  const upload = $("mode-upload");

  const setMode = (mode) => {
    const isUpload = mode === "upload";
    $("check-form").hidden = isUpload;
    $("panel-upload").hidden = !isUpload;
    $("result-card").hidden = true;
    paste.classList.toggle("is-active", !isUpload);
    upload.classList.toggle("is-active", isUpload);
    paste.setAttribute("aria-pressed", String(!isUpload));
    upload.setAttribute("aria-pressed", String(isUpload));
  };

  paste.addEventListener("click", () => setMode("paste"));
  upload.addEventListener("click", () => setMode("upload"));
}

function bindUpload() {
  const input = $("doc-file");
  const zone = $("doc-dropzone");

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) uploadDocument(input.files[0]);
  });

  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("is-over");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) uploadDocument(file);
  });

  /* A file dropped outside the zone must not navigate the page away. */
  ["dragover", "drop"].forEach((name) => {
    document.addEventListener(name, (event) => {
      const types = event.dataTransfer ? event.dataTransfer.types : [];
      if (types && Array.prototype.includes.call(types, "Files")) event.preventDefault();
    });
  });

  $("doc-ask").addEventListener("click", runAsk);
  $("doc-clear").addEventListener("click", () => {
    state.doc = null;
    input.value = "";
    $("doc-query").value = "";
    $("doc-status").hidden = true;
    $("doc-answer").hidden = true;
    $("doc-error").hidden = true;
    $("result-card").hidden = true;
  });
}

async function uploadDocument(file) {
  const errorNode = $("doc-error");
  const status = $("doc-status");
  const ask = $("doc-ask");

  if (!DOC_EXT.test(file.name) && file.type !== "text/plain") {
    showError(errorNode, "Choose a PDF or a text file.");
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    showError(errorNode, `That file is larger than the ${MAX_UPLOAD_MB} MB limit.`);
    return;
  }

  errorNode.hidden = true;
  status.hidden = false;
  status.textContent = `Reading ${file.name}. The first upload also loads the embedding model, so it can take a few seconds.`;
  $("doc-answer").hidden = true;
  $("result-card").hidden = true;
  ask.disabled = true;

  try {
    state.doc = await API.upload("/documents/upload", file);
    const kind = state.doc.kind === "pdf" ? `${state.doc.pages} pages` : "text file";
    status.textContent =
      `${state.doc.filename} is loaded: ${kind}, ${state.doc.passages} passages.`;
    $("doc-query").focus();
  } catch (err) {
    state.doc = null;
    status.hidden = true;
    showError(errorNode, err.message);
  } finally {
    ask.disabled = false;
  }
}

async function runAsk() {
  if (state.busy) return;
  const errorNode = $("doc-error");
  const query = $("doc-query").value.trim();

  if (!state.doc) {
    showError(errorNode, "Upload a document first.");
    return;
  }
  if (!query) {
    showError(errorNode, "Type a question about the document.");
    return;
  }
  errorNode.hidden = true;

  state.busy = true;
  const ask = $("doc-ask");
  const label = ask.textContent;
  ask.disabled = true;
  ask.textContent = "Asking";
  $("doc-answer").hidden = true;
  $("result-card").hidden = false;
  $("result-skeleton").hidden = false;
  $("result-body").hidden = true;
  $("result-latency").textContent = "";

  try {
    const reply = await API.post("/ask", { doc_id: state.doc.doc_id, query });
    renderDocumentAnswer(reply);
    if (!reply.verdict) {
      renderNoVerdict(reply);
    } else if (reply.mode === "extractive" && reply.grounding === "exact") {
      renderGrounded(reply);
    } else {
      renderResult(reply.verdict, { fromDocument: true, note: reply.note });
    }
  } catch (err) {
    $("result-card").hidden = true;
    showError(errorNode, err.message);
  } finally {
    state.busy = false;
    ask.disabled = false;
    ask.textContent = label;
    $("result-skeleton").hidden = true;
  }
}

function renderDocumentAnswer(reply) {
  $("doc-answer").hidden = false;
  $("doc-answer-text").textContent =
    reply.answer || "The document does not appear to answer this question.";
  $("doc-answer-mode").textContent =
    reply.mode === "llm"
      ? "written by the model"
      : reply.mode === "extractive"
        ? "quoted from the document"
        : "no answer found";
  $("doc-answer-source").textContent = reply.grounded_in || "";
  $("doc-source").hidden = !reply.grounded_in;
}

function renderNoVerdict(reply) {
  $("result-body").hidden = false;
  const verdict = $("result-verdict");
  verdict.classList.remove("is-approved");
  verdict.classList.add("is-withheld");
  useIcon($("result-icon"), "withhold");
  $("result-verdict-text").textContent = "No answer";
  $("result-meaning").textContent = reply.note;
  $("result-band").hidden = true;
  $("result-score").textContent = "n/a";
  $("result-threshold").textContent = "n/a";
  $("result-domain").textContent = "n/a";
  $("result-latency").textContent = "";
  $("result-note").textContent = "";
}

function renderGrounded(reply) {
  const verdict = reply.verdict;
  $("result-body").hidden = false;
  const node = $("result-verdict");
  node.classList.remove("is-withheld");
  node.classList.add("is-approved");
  useIcon($("result-icon"), "check");
  $("result-verdict-text").textContent = "Grounded";
  $("result-meaning").textContent =
    "This answer appears word for word in the source passage, so it is grounded by construction. " +
    "The conformal verdict is for answers a model writes; set a provider key to have the system write one.";
  $("result-band").hidden = true;
  $("fact-score").hidden = false;
  $("result-score").textContent = score(verdict.risk_score);
  $("result-threshold").textContent = score(verdict.threshold);
  $("result-domain").textContent = verdict.domain;
  $("result-latency").textContent = `${Math.round(verdict.latency_ms)} ms`;
  $("result-note").textContent =
    `Exact quote. The checker reads it at ${score(verdict.risk_score)} against a threshold of ${score(verdict.threshold)}; ` +
    "containment is used instead because the sentence is present in the passage.";
  state.toleranceTopic = verdict.domain;
  renderTolerance();
}

function topicTolerance(name) {
  const found = state.domains.find((d) => d.name === name);
  return found ? found.alpha : (state.config ? state.config.alpha : 0.15);
}

/* ------------------------------------------------ check: tolerance card */

function renderAlphaRow(container, active, onPick) {
  container.replaceChildren();
  ALPHA_CHOICES.forEach((value) => {
    const chip = el("button", {
      type: "button",
      class: `chip button-xsm${Math.abs(value - active) < 1e-9 ? " is-active" : ""}`,
      "aria-pressed": String(Math.abs(value - active) < 1e-9),
    }, alpha(value));
    chip.addEventListener("click", () => onPick(value));
    container.appendChild(chip);
  });
}

function renderTolerance() {
  const topic = state.toleranceTopic || "general";
  const cfg = state.config || {};
  if (cfg.alpha === undefined) return;

  let alphaValue;
  let thresholdValue;
  let samples;
  let needed;
  let calibrated;
  let certifiable;

  if (topic === "general") {
    const cal = state.calibration || {};
    alphaValue = cfg.alpha;
    thresholdValue = cal.threshold;
    samples = cal.n_samples;
    needed = cal.min_samples_required;
    calibrated = cal.calibrated === true;
    certifiable = cal.certifiable === true;
  } else {
    const domain = state.domains.find((d) => d.name === topic);
    if (!domain) return;
    alphaValue = domain.alpha;
    thresholdValue = domain.threshold;
    samples = domain.n_samples;
    needed = domain.min_samples_required;
    calibrated = domain.threshold !== null && domain.threshold !== undefined;
    certifiable = domain.certifiable === true;
  }

  renderToleranceTopicRow();
  renderAlphaRow($("alpha-row"), alphaValue, async (value) => {
    try {
      const body = topic === "general" ? { alpha: value, online: true } : { alpha: value, domain: topic };
      await API.post("/config", body);
      await loadConfig();
      await loadCalibration();
      await loadDomains();
      renderTolerance();
      if (!$("result-card").hidden && $("f-response").value.trim()) runCheck();
    } catch (err) {
      showError($("check-error"), err.message);
    }
  });

  $("tolerance-alpha").textContent = alpha(alphaValue);
  $("tolerance-topic-label").textContent = title(topic);
  $("tolerance-threshold").textContent = calibrated ? score(thresholdValue) : "n/a";
  $("tolerance-samples").textContent = calibrated ? String(samples) : "0";
  $("tolerance-needed").textContent = calibrated || needed ? String(needed) : "n/a";

  const note = $("tolerance-note");
  if (!calibrated) {
    note.textContent = `The ${topic} topic has no calibration yet. Run the seeding step, then reload this page.`;
    note.classList.add("is-short");
    return;
  }
  const short = samples < needed;
  note.textContent = short
    ? `Only ${samples} calibration samples are loaded but this tolerance needs ${needed}, so the threshold sits above every score and everything is withheld. Raise the tolerance or add samples.`
    : `Certified: ${samples} calibration samples is enough for a tolerance of ${alpha(alphaValue)}. Lowering the tolerance raises the sample count it needs.`;
  $("promise-text").textContent =
    `Promise: of the answers approved, the expected share that are wrong stays at or below ${alpha(alphaValue)}.`;
  $("tolerance-note").classList.toggle("is-short", short || !certifiable);
}

function renderToleranceTopicRow() {
  const row = $("tolerance-topic-row");
  if (!row || !state.domains.length) return;
  row.replaceChildren();
  state.domains.forEach((domain) => {
    const on = domain.name === (state.toleranceTopic || "general");
    const chip = el("button", {
      type: "button",
      class: `chip button-xsm${on ? " is-active" : ""}`,
      "aria-pressed": String(on),
    }, title(domain.name));
    chip.addEventListener("click", () => {
      state.toleranceTopic = domain.name;
      renderTolerance();
    });
    row.appendChild(chip);
  });
  row.style.marginBottom = "var(--s-100)";
}

/* -------------------------------------------------------------- topics */

function renderDomains() {
  const grid = $("domain-grid");
  grid.replaceChildren();

  const order = ["general", "medicine", "finance"];
  const list = state.domains
    .slice()
    .sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));

  list.forEach((domain) => {
    const card = el("section", { class: "card" });
    const head = el("div", { class: "card-head" });
    head.appendChild(el("h4", { class: "title-1" }, title(domain.name)));
    head.appendChild(el("span", { class: "caption-4 card-note" },
      domain.certifiable ? "certified" : "needs more samples"));
    card.appendChild(head);

    card.appendChild(el("p", { class: "body-6 card-body" },
      TOPIC_BLURB[domain.name] || "Its own calibration set and its own threshold."));

    const facts = el("dl", { class: "facts" });
    const rows = [
      ["Tolerance", `\u03b1 = ${alpha(domain.alpha)}`],
      ["Threshold", domain.threshold === null ? "n/a" : score(domain.threshold)],
      ["Samples", String(domain.n_samples || 0)],
      ["Approved", domain.stats ? pct(domain.stats.coverage) : "n/a"],
    ];
    rows.forEach(([label, value]) => {
      const wrap = el("div", { class: "fact" });
      wrap.appendChild(el("dt", { class: "caption-4" }, label));
      wrap.appendChild(el("dd", { class: "num" }, value));
      facts.appendChild(wrap);
    });
    card.appendChild(facts);

    const risk = el("p", { class: "caption-2 card-body" });
    if (domain.stats) {
      risk.textContent =
        `In sample, ${pct(domain.stats.approved_wrong_rate)} of answers were approved and wrong, ` +
        `against a tolerance of ${alpha(domain.alpha)}.`;
    }
    risk.style.marginTop = "var(--s-200)";
    card.appendChild(risk);

    const alphaLabel = el("span", { class: "title-4" }, "Change the tolerance");
    alphaLabel.style.display = "block";
    alphaLabel.style.marginTop = "auto";
    alphaLabel.style.paddingTop = "var(--s-300)";
    alphaLabel.style.marginBottom = "var(--s-75)";
    card.appendChild(alphaLabel);

    const alphaRow = el("div", { class: "chip-row" });
    renderAlphaRow(alphaRow, domain.alpha, async (value) => {
      try {
        await API.post("/config", { alpha: value, domain: domain.name });
        await loadDomains();
        await loadCalibration();
      } catch (err) {
        console.warn(err.message);
      }
    });
    card.appendChild(alphaRow);

    const caseRow = el("div", { class: "chip-row case-row" });
    const spec = state.demo ? state.demo.showcase[domain.name] : null;
    if (spec) {
      [
        ["faithful", "Faithful answer"],
        ["subtle", "Subtly wrong answer"],
      ].forEach(([kind, label]) => {
        const chip = el("button", { type: "button", class: "chip button-xsm" }, label);
        chip.addEventListener("click", () => runDomainCase(domain.name, kind, chip));
        caseRow.appendChild(chip);
      });
    }
    card.appendChild(caseRow);
    card.appendChild(el("p", { class: "caption-2 card-foot", "data-case-out": domain.name }, ""));

    grid.appendChild(card);
  });

  renderMinTable();
  renderTolerance();
}

async function runDomainCase(domain, kind, chip) {
  const spec = state.demo.showcase[domain];
  const item = spec.cases.find((c) => c.kind === kind);
  const out = document.querySelector(`[data-case-out="${domain}"]`);
  const label = chip.textContent;
  chip.disabled = true;
  chip.textContent = "Running";
  out.textContent = "";
  try {
    const result = await API.post("/predict", {
      query: spec.query,
      context: [spec.context],
      response: item.response,
      domain,
    });
    out.textContent =
      `${result.abstained ? "Withheld" : "Approved"} at a score of ${score(result.risk_score)}, ` +
      `against a ${domain} threshold of ${score(result.threshold)}.`;
  } catch (err) {
    out.textContent = err.message;
  } finally {
    chip.disabled = false;
    chip.textContent = label;
  }
}

function renderMinTable() {
  const body = $("min-body");
  if (!body) return;
  body.replaceChildren();
  MIN_SAMPLES.forEach((row) => {
    const tr = el("tr");
    tr.appendChild(el("td", { class: "num cell-name" }, alpha(row.alpha)));
    tr.appendChild(el("td", { class: "num" }, String(row.n)));
    tr.appendChild(el("td", {}, `A calibration set of ${row.n} rows can certify this tolerance.`));
    body.appendChild(tr);
  });
}

function renderTopicRow() {
  if (!state.domains.length || state.activeKind) return;
  $("check-domain-note").textContent =
    `Auto-routes by topic, over ${state.domains.length} loaded`;
}

function renderChartTopicRow() {
  const row = $("chart-domain-row");
  if (!row) return;
  row.replaceChildren();
  state.domains.forEach((domain) => {
    const chip = el("button", {
      type: "button",
      class: `chip button-xsm${domain.name === state.chartTopic ? " is-active" : ""}`,
      "aria-pressed": String(domain.name === state.chartTopic),
    }, title(domain.name));
    chip.addEventListener("click", () => {
      state.chartTopic = domain.name;
      renderChartTopicRow();
      renderSeparation();
    });
    row.appendChild(chip);
  });
}

/* -------------------------------------------------------------- status */

function renderMetrics() {
  const grid = $("metric-grid");
  if (!grid) return;
  grid.replaceChildren();

  const metrics = state.metrics;
  const cal = state.calibration;
  const tolerance = state.config ? state.config.alpha : 0.15;
  const traffic = metrics ? metrics.total_predictions : 0;
  const labelled = metrics ? metrics.labeled_feedback : 0;

  const cards = [
    {
      label: "Approved and wrong",
      value: labelled ? pct(metrics.approved_wrong_rate) : "n/a",
      note: labelled
        ? `The promise, measured against a tolerance of ${alpha(tolerance)}.`
        : "Fills in once labelled feedback arrives. The promise itself is set on the calibration set.",
      tone: labelled && metrics.approved_wrong_rate > tolerance ? "is-caution" : labelled ? "is-good" : "",
    },
    {
      label: "False alarms",
      value: labelled ? pct(metrics.false_alarm_rate) : "n/a",
      note: "Of the answers withheld, the share that were actually fine. Not part of the promise.",
      tone: "",
    },
    {
      label: "Approved",
      value: traffic ? pct(metrics.coverage) : "n/a",
      note: traffic
        ? "Share of answers the system was willing to approve."
        : "No answers checked since the server started.",
      tone: "",
    },
    {
      label: "Withheld",
      value: traffic ? pct(metrics.abstention_rate) : "n/a",
      note: traffic ? `${traffic} answers checked since start.` : "No answers checked since the server started.",
      tone: "",
    },
  ];

  cards.forEach((card) => {
    const node = el("div", { class: "metric" });
    node.appendChild(el("p", { class: "caption-4 metric-label" }, card.label));
    node.appendChild(el("p", { class: `num metric-value ${card.tone}` }, card.value));
    node.appendChild(el("p", { class: "caption-2 metric-note" }, card.note));
    grid.appendChild(node);
  });

  const foot = el("p", { class: "caption-2 metric-foot" });
  foot.textContent = cal && cal.calibrated
    ? `${labelled} labelled answers so far. The two rates above describe traffic; the promise is certified on ${cal.n_samples} calibration samples.`
    : "Waiting for the calibration summary from the server.";
  document.querySelectorAll(".metric-foot").forEach((node) => node.remove());
  grid.parentElement.insertBefore(foot, $("separation-card"));
}

function renderDriftTable() {
  const body = $("drift-body");
  if (!body) return;
  body.replaceChildren();

  if (!state.domains.length) {
    const tr = el("tr");
    const td = el("td", { class: "body-6" }, "No topics are registered yet.");
    td.colSpan = 5;
    tr.appendChild(td);
    body.appendChild(tr);
    $("drift-note").textContent = "";
    return;
  }

  state.domains.forEach((domain) => {
    const m = domain.metrics || {};
    const drifted = domain.drift && domain.drift.drifted;
    const seen = (m.total_predictions || 0) > 0;
    const labelled = (m.labeled_feedback || 0) > 0;
    const tr = el("tr");
    tr.appendChild(el("td", { class: "cell-name" }, title(domain.name)));
    tr.appendChild(el("td", { class: "num" }, alpha(domain.alpha)));
    tr.appendChild(el("td", { class: "num" }, labelled ? pct(m.approved_wrong_rate || 0) : "n/a"));
    tr.appendChild(el("td", { class: "num" }, seen ? pct(m.abstention_rate || 0) : "n/a"));
    const stateCell = el("td");
    const pill = el("span", { class: `pill ${drifted ? "pill-warn" : "pill-ok"}` },
      drifted ? "Drift flagged" : "In range");
    stateCell.appendChild(pill);
    tr.appendChild(stateCell);
    body.appendChild(tr);
  });

  const labels = state.domains.reduce((sum, d) => sum + ((d.metrics || {}).labeled_feedback || 0), 0);
  $("drift-note").textContent = labels
    ? `${labels} labelled answers so far`
    : "no labelled answers yet";
}

async function renderSeparation() {
  const wrap = $("separation-chart");
  if (!wrap) return;

  let data;
  try {
    data = await API.get(`/calibration/scores?domain=${encodeURIComponent(state.chartTopic)}`);
  } catch (err) {
    wrap.replaceChildren(el("p", { class: "body-6" }, err.message));
    return;
  }

  const points = data.points || [];
  if (!points.length) {
    wrap.replaceChildren(el("p", { class: "body-6" }, "No calibration points for this topic."));
    $("sep-note").textContent = "calibration set";
    return;
  }

  $("sep-note").textContent = data.calibrated
    ? `calibration set \u00b7 threshold ${score(data.threshold)}`
    : "calibration set \u00b7 not certifiable";

  const width = 640;
  const padLeft = 96;
  const padRight = 16;
  const rowTop = 32;
  const rowGap = 56;
  const height = rowTop + rowGap + 48;
  const plotWidth = width - padLeft - padRight;
  const x = (value) => padLeft + clamp01(value) * plotWidth;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${points.length} calibration answers plotted by score. The threshold sits at ${score(data.threshold)}.`,
  });

  const rows = [
    { label: "faithful", value: 0, top: rowTop, color: "var(--ok)" },
    { label: "wrong", value: 1, top: rowTop + rowGap, color: "var(--bad)" },
  ];

  rows.forEach((row) => {
    svg.appendChild(svgEl("line", {
      x1: padLeft, x2: padLeft + plotWidth, y1: row.top, y2: row.top,
      stroke: "var(--line)", "stroke-width": 1,
    }));
    svg.appendChild(svgEl("text", {
      x: padLeft - 12, y: row.top + 4, "text-anchor": "end",
      fill: "var(--text-3)", "font-size": 12, "font-family": "Geist, sans-serif",
    }, row.label));
  });

  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const tx = x(tick);
    svg.appendChild(svgEl("line", {
      x1: tx, x2: tx, y1: rowTop - 12, y2: rowTop + rowGap + 12,
      stroke: "var(--line)", "stroke-width": 1,
    }));
    svg.appendChild(svgEl("text", {
      x: tx, y: rowTop + rowGap + 36, "text-anchor": "middle",
      fill: "var(--text-3)", "font-size": 12, "font-family": "Geist Mono, monospace",
    }, tick.toFixed(2)));
  });

  const cut = data.threshold === null ? 1 : data.threshold;
  const thresholdX = x(cut);
  svg.appendChild(svgEl("line", {
    x1: thresholdX, x2: thresholdX, y1: rowTop - 22, y2: rowTop + rowGap + 18,
    stroke: "var(--text-2)", "stroke-width": 1, "stroke-dasharray": "4 4",
  }));

  rows.forEach((row) => {
    const inRow = points
      .filter((point) => point.label === row.value)
      .sort((a, b) => a.score - b.score);
    inRow.forEach((point, index) => {
      const stack = index % 3;
      svg.appendChild(svgEl("circle", {
        cx: x(point.score),
        cy: row.top - 8 + stack * 8,
        r: 4,
        fill: row.color,
        opacity: 0.85,
      }));
    });
  });

  wrap.replaceChildren(svg);
}

/* -------------------------------------------------------------- method */

function renderFormula() {
  if (!state.config) return;
  $("formula-line").textContent =
    `E[ P(approved and wrong) ] \u2264 ${alpha(state.config.alpha)}`;
}

/* --------------------------------------------------------- chrome text */

function renderTopbarState() {
  if (!state.config) return;
  const parts = [];
  parts.push(state.config.calibrated ? "calibrated" : "not calibrated");
  parts.push(`${state.config.documents} documents`);
  $("topbar-state").textContent = parts.join(" \u00b7 ");
}

function renderFooter() {
  if (!state.config) return;
  const model =
    state.config.nli_model === "small"
      ? "nli-deberta-v3-small"
      : state.config.nli_model === "large"
        ? "nli-deberta-v3-large"
        : state.config.nli_model;
  const judge = state.config.llm_judge_configured
    ? "judge on"
    : "judge off, no provider key";
  $("footer-state").textContent =
    `Local demo \u00b7 ${state.config.documents} documents \u00b7 e5-base-v2 retrieval \u00b7 ` +
    `nli-deberta scoring (${model}) \u00b7 ${judge}.`;
}

/* ------------------------------------------------------------------ go */

async function boot() {
  bindNav();
  bindCheckForm();
  bindModeSwitch();
  bindUpload();
  activate("check", false);

  const failures = [];
  const step = async (label, fn) => {
    try {
      await fn();
    } catch (err) {
      failures.push(`${label}: ${err.message}`);
    }
  };

  await step("config", loadConfig);
  await step("demo", loadDemo);
  await step("calibration", loadCalibration);
  await step("topics", loadDomains);
  await step("status", loadStatus);

  if (failures.length) {
    showError($("check-error"), failures.join(" "));
  }
}

document.addEventListener("DOMContentLoaded", boot);
