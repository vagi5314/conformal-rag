/* app.js — view routing, state, and plain-language rendering */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---------- helpers ----------

  let toastTimer = null;
  function toast(message, kind = "") {
    const el = $("toast");
    el.textContent = message;
    el.className = `toast ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 3500);
  }

  const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
  const pct0 = (x) => `${(x * 100).toFixed(0)}%`;

  function busy(btn, on, label) {
    if (on) {
      btn.dataset.origLabel = btn.textContent;
      btn.textContent = label || "Working\u2026";
      btn.disabled = true;
    } else {
      btn.textContent = btn.dataset.origLabel || btn.textContent;
      btn.disabled = false;
    }
  }

  function strictness(alpha) {
    if (alpha <= 0.005) return "Very strict";
    if (alpha <= 0.02) return "Strict";
    if (alpha <= 0.05) return "Standard";
    return "Relaxed";
  }

  // ---------- view routing ----------

  const viewHooks = {};
  let statusTimer = null;

  function switchView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    $(`view-${name}`).classList.add("active");
    document.querySelector(`.nav-btn[data-view="${name}"]`).classList.add("active");
    if (name !== "status") clearInterval(statusTimer);
    if (viewHooks[name]) viewHooks[name]();
  }

  $("nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-btn");
    if (btn) switchView(btn.dataset.view);
  });

  // ---------- check an answer ----------

  let docCount = 0;
  let currentAlpha = 0.05;

  async function refreshEngineStatus() {
    try {
      const cfg = await API.getConfig();
      currentAlpha = cfg.alpha;
      $("alpha-slider").value = cfg.alpha;
      $("alpha-value").textContent = pct(cfg.alpha);
      $("online-toggle").checked = cfg.online;
      docCount = cfg.documents;
      $("index-status").textContent =
        docCount > 0 ? `${docCount} sources saved` : "no sources saved yet";
      const taught = cfg.calibrated
        ? "taught"
        : "NOT taught yet \u2014 verdicts use a default line, no guarantee";
      $("engine-status").textContent =
        `Tolerance ${pct(cfg.alpha)} \u00b7 auto-adjust ${cfg.online ? "on" : "off"} \u00b7 ${taught}`;
      $("engine-status").className = `status-line ${cfg.calibrated ? "ok" : "err"}`;
    } catch (err) {
      $("engine-status").textContent = `Can't reach the server: ${err.message}`;
      $("engine-status").className = "status-line err";
    }
  }

  $("alpha-slider").addEventListener("input", (e) => {
    $("alpha-value").textContent = pct(parseFloat(e.target.value));
  });

  $("init-btn").addEventListener("click", async () => {
    const btn = $("init-btn");
    busy(btn, true, "Applying\u2026");
    try {
      const alpha = parseFloat($("alpha-slider").value);
      const online = $("online-toggle").checked;
      const cfg = await API.setConfig(alpha, online);
      currentAlpha = cfg.alpha;
      $("engine-status").textContent =
        `Tolerance ${pct(cfg.alpha)} \u00b7 auto-adjust ${cfg.online ? "on" : "off"} \u00b7 ready`;
      $("engine-status").className = "status-line ok";
      toast(`Settings applied: ${pct(cfg.alpha)} tolerance, auto-adjust ${cfg.online ? "on" : "off"}`, "ok");
    } catch (err) {
      toast(`Couldn't apply settings: ${err.message}`, "err");
    } finally {
      busy(btn, false);
    }
  });

  $("load-samples-btn").addEventListener("click", async () => {
    try {
      const { documents } = await API.samples();
      $("docs-input").value = documents.join("\n");
      toast(`${documents.length} example sources loaded \u2014 press "Save sources"`, "ok");
    } catch (err) {
      toast(`Couldn't load examples: ${err.message}`, "err");
    }
  });

  $("index-btn").addEventListener("click", async () => {
    const btn = $("index-btn");
    const lines = $("docs-input").value.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) { toast("Nothing to save \u2014 add some sources first", "err"); return; }
    busy(btn, true, "Saving\u2026");
    try {
      const { indexed } = await API.index(lines);
      docCount = indexed;
      $("index-status").textContent = `${indexed} sources saved`;
      toast(`Saved ${indexed} sources`, "ok");
    } catch (err) {
      toast(`Couldn't save sources: ${err.message}`, "err");
    } finally {
      busy(btn, false);
    }
  });

  async function currentThreshold() {
    try {
      const cal = await API.calibration();
      if (cal.calibrated) return cal.threshold;
      const online = await API.online();
      if (online.enabled) return online.threshold;
    } catch (_) { /* fall through */ }
    return 0.5;
  }

  function renderVerdict(result, threshold) {
    const verdict = $("verdict-card");
    const score = result.risk_score;

    if (result.abstained) {
      verdict.className = "verdict abstained";
      verdict.innerHTML =
        `<div class="verdict-word">\u2717 Not sure</div>` +
        `<div class="verdict-detail">We can't confirm this answer is supported by the sources ` +
        `(only ${pct(score)} support). To keep its promise, the system would rather refuse than guess. ` +
        `Better not to use this answer.</div>`;
    } else {
      verdict.className = "verdict accepted";
      verdict.innerHTML =
        `<div class="verdict-word">\u2713 Trustworthy</div>` +
        `<div class="verdict-detail">This answer is supported by the sources ` +
        `(${pct(score)} support). Under your ${pct(result.risk_bound)} tolerance, ` +
        `across many answers at most ${pct(result.risk_bound)} of the ones we approve ` +
        `will be wrong. We do not promise anything about this single answer.</div>`;
    }
    if (!result.calibrated) {
      verdict.innerHTML +=
        `<div class="verdict-detail" style="margin-top:10px;color:var(--ember)">` +
        `\u26a0 Not taught yet \u2014 this verdict uses a rough default line and carries ` +
        `no statistical guarantee. Visit \u201cTeach It\u201d to make the promise real.</div>`;
    }

    // dial: score is "how supported" — higher is better
    const ARC = 257.6; // semicircle arc length, r = 82
    $("dial-block").className = `dial-wrap ${result.abstained ? "bad" : "good"}`;
    $("dial-arc").style.strokeDashoffset = `${ARC * (1 - Math.max(0.01, score))}`;
    $("dial-value").textContent = pct(score);
    // threshold tick on the semicircle: 0 = left end, 1 = right end
    const a = Math.PI * threshold;
    const tick = $("dial-tick");
    tick.setAttribute("x1", (100 - 72 * Math.cos(a)).toFixed(1));
    tick.setAttribute("y1", (108 - 72 * Math.sin(a)).toFixed(1));
    tick.setAttribute("x2", (100 - 92 * Math.cos(a)).toFixed(1));
    tick.setAttribute("y2", (108 - 92 * Math.sin(a)).toFixed(1));
    $("dial-caption").textContent =
      `The arc shows how strongly the sources support the answer. ` +
      `The tick (${pct(threshold)}) is where the system draws the line: ` +
      `below it means "not sure".`;

    // how we decided — plain language
    const b = result.scoring_breakdown;
    const lines = [];
    lines.push(`<span class="k">fast checker:</span> <span class="v">found ${pct(b.nli_score)} support in the sources</span>`);
    if (b.used_llm_judge && b.llm_judge_score != null) {
      lines.push(`<span class="k">careful judge:</span> <span class="v">the fast checker wasn't confident, so we double-checked \u2014 it found ${pct(b.llm_judge_score)} support</span>`);
    } else {
      lines.push(`<span class="k">careful judge:</span> <span class="v">not needed \u2014 the fast checker was confident</span>`);
    }
    lines.push(`<span class="k">final verdict:</span> <span class="v ${result.abstained ? "bad" : "good"}">${pct(b.final_score)} supported</span>`);
    const ci = result.confidence_interval;
    lines.push(`<span class="k">statistical floor:</span> <span class="v">we're confident support is at least ${pct(ci[0])}</span>`);
    lines.push(`<span class="k">time taken:</span> <span class="v">${(result.latency_ms / 1000).toFixed(1)}s</span>`);
    $("how-decided").innerHTML =
      `<div style="font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin-bottom:8px">how we decided</div>` +
      lines.join("<br />");
  }

  $("predict-btn").addEventListener("click", async () => {
    const btn = $("predict-btn");
    const query = $("query-input").value.trim();
    const response = $("response-input").value.trim();
    if (!query) { toast("Add the question first", "err"); return; }
    if (!response) { toast("Paste the AI's answer so we can check it", "err"); return; }

    busy(btn, true, "Checking\u2026 (first check loads the model)");
    try {
      let context = $("context-input").value.split("\n").map((l) => l.trim()).filter(Boolean);

      if ($("auto-retrieve").checked && docCount > 0 && !context.length) {
        const { results } = await API.retrieve(query, 5);
        if (results.length) context = results;
      }

      const [result, threshold] = await Promise.all([
        API.predict(query, context, response),
        currentThreshold(),
      ]);

      renderVerdict(result, threshold);
      $("predict-result").classList.remove("hidden");
    } catch (err) {
      toast(`Check failed: ${err.message}`, "err");
    } finally {
      busy(btn, false);
    }
  });

  // ---------- teach it ----------

  const EXAMPLE = {
    queries: [
      "What is the capital of France?",
      "What is the capital of France?",
      "Who created Python?",
      "Who created Python?",
      "What is conformal prediction?",
    ],
    contexts: [
      "The capital of France is Paris.",
      "The capital of France is Paris.",
      "Python was created by Guido van Rossum in 1991.",
      "Python was created by Guido van Rossum in 1991.",
      "Conformal prediction gives distribution-free coverage guarantees.",
    ],
    responses: [
      "The capital of France is Paris.",
      "The capital of France is Lyon.",
      "Python was created by Guido van Rossum.",
      "Python was created by Google in 2020.",
      "It is a framework with guaranteed coverage.",
    ],
    labels: ["0", "1", "0", "1", "0"],
  };

  $("fill-example-btn").addEventListener("click", () => {
    $("cal-queries").value = EXAMPLE.queries.join("\n");
    $("cal-contexts").value = EXAMPLE.contexts.join("\n");
    $("cal-responses").value = EXAMPLE.responses.join("\n");
    $("cal-labels").value = EXAMPLE.labels.join("\n");
    toast("Worked example loaded \u2014 press \u201cTeach it\u201d", "ok");
  });

  function parseLines(id) {
    return $(id).value.split("\n").map((l) => l.trim()).filter(Boolean);
  }

  $("calibrate-btn").addEventListener("click", async () => {
    const btn = $("calibrate-btn");
    const queries = parseLines("cal-queries");
    const contexts = parseLines("cal-contexts");
    const responses = parseLines("cal-responses");
    const labels = parseLines("cal-labels");

    const n = queries.length;
    if (!n) { toast("Add some examples first", "err"); return; }
    if (contexts.length !== n || responses.length !== n || labels.length !== n) {
      $("calibrate-status").textContent =
        `The four boxes must have the same number of lines (currently ${n}, ${contexts.length}, ${responses.length}, ${labels.length}).`;
      $("calibrate-status").className = "status-line inline err";
      return;
    }
    const parsedLabels = labels.map((l) => parseInt(l, 10));
    if (parsedLabels.some((l) => l !== 0 && l !== 1)) {
      $("calibrate-status").textContent = "Labels must be 0 (correct) or 1 (made up).";
      $("calibrate-status").className = "status-line inline err";
      return;
    }

    const data = queries.map((q, i) => ({
      query: q, context: contexts[i], response: responses[i], label: parsedLabels[i],
    }));

    busy(btn, true, "Teaching\u2026");
    try {
      const stats = await API.calibrate(data);
      $("calibrate-status").textContent = `Learned from ${stats.n_samples} examples.`;
      $("calibrate-status").className = "status-line inline ok";
      const warningLines = (stats.warnings || []).map(
        (w) => `<span class="v bad">\u26a0 ${w}</span><br />`
      ).join("");
      $("calibrate-result").innerHTML =
        `<div style="font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin-bottom:8px">what it learned</div>` +
        warningLines +
        `It studied <span class="v">${stats.n_samples} examples</span> (${stats.n_hallucinations} of them made up).<br />` +
        `It will now say <span class="v">"not sure"</span> for any answer scoring below <span class="v good">${pct(stats.threshold)}</span> support.<br />` +
        `On these examples, it would have approved <span class="v">${pct(stats.coverage)}</span> of them ` +
        `\u2014 and of the ones it approved, <span class="v ${stats.estimated_fdr > 0 ? "bad" : "good"}">${pct(stats.estimated_fdr)}</span> were actually wrong.<br />` +
        `Your tolerance: <span class="v">${pct(stats.alpha)}</span>.`;
      $("calibrate-result").classList.remove("hidden");
      toast(`Taught on ${stats.n_samples} examples \u2014 the line is now at ${pct(stats.threshold)} support`, "ok");
    } catch (err) {
      $("calibrate-status").textContent = `Couldn't teach it: ${err.message}`;
      $("calibrate-status").className = "status-line inline err";
    } finally {
      busy(btn, false);
    }
  });

  // ---------- live status ----------

  const history = { fdr: [], coverage: [], threshold: [] };
  const MAX_POINTS = 100;

  async function pollStatus() {
    try {
      const [metrics, online, calibration] = await Promise.all([
        API.metrics(), API.online(), API.calibration(),
      ]);

      $("dash-total").textContent = metrics.total_predictions;

      const fdrEl = $("dash-fdr");
      fdrEl.textContent = pct(metrics.fdr);
      fdrEl.className = `stat-value ${metrics.fdr > currentAlpha ? "warn" : "good"}`;

      $("dash-coverage").textContent = pct0(metrics.coverage);
      $("dash-abstention").textContent = pct0(metrics.abstention_rate);

      history.fdr.push(metrics.fdr);
      history.coverage.push(metrics.coverage);
      for (const key of Object.keys(history)) {
        if (history[key].length > MAX_POINTS) history[key].shift();
      }

      Charts.render($("chart-fdr"), history.fdr, {
        min: 0,
        max: Math.max(0.1, ...history.fdr, currentAlpha * 1.5),
        baseline: currentAlpha,
        baselineLabel: `your tolerance (${pct(currentAlpha)})`,
        color: "terra",
        yFormat: (v) => pct0(v),
      });
      Charts.render($("chart-coverage"), history.coverage, {
        min: 0, max: 1, color: "green", yFormat: (v) => pct0(v),
      });

      if (online.enabled) {
        history.threshold.push(online.threshold);
        if (history.threshold.length > MAX_POINTS) history.threshold.shift();
        $("aci-panel").innerHTML =
          `<span class="k">current line:</span> <span class="v good">${pct(online.threshold)} support</span><br />` +
          `<span class="k">feedback seen:</span> <span class="v">${online.n_updates} reports</span><br />` +
          `<span class="k">memory:</span> <span class="v">last ${online.window_size} scores</span><br />` +
          `<span class="k">learning pace:</span> <span class="v">${online.level.toFixed(3)}</span>`;
        Charts.render($("chart-threshold"), history.threshold, {
          min: 0, max: 1, color: "blue", yFormat: (v) => pct0(v),
        });
      } else {
        $("aci-panel").textContent = "Auto-adjust is off \u2014 the line stays where teaching put it. Turn it on in \u201cCheck an Answer\u201d \u2192 Setup.";
        Charts.render($("chart-threshold"), [], {});
      }

      if (calibration.calibrated) {
        $("dash-calibration").innerHTML =
          `<span class="k">taught on:</span> <span class="v">${calibration.n_samples} examples</span><br />` +
          `<span class="k">the line:</span> <span class="v good">${pct(calibration.threshold)} support</span><br />` +
          `<span class="k">on training data:</span> <span class="v">${pct(calibration.estimated_fdr)} of approved answers were wrong</span><br />` +
          `<span class="k">tolerance:</span> <span class="v">${pct(calibration.alpha)}</span>`;
      } else {
        $("dash-calibration").textContent =
          "It hasn't been taught yet \u2014 it's using a rough default standard. Visit \u201cTeach It\u201d to sharpen it.";
      }
    } catch (_) { /* server down; keep last render */ }
  }

  viewHooks.status = () => {
    pollStatus();
    clearInterval(statusTimer);
    statusTimer = setInterval(pollStatus, 3000);
  };

  $("record-btn").addEventListener("click", async () => {
    const btn = $("record-btn");
    const query = $("mon-query").value.trim();
    const response = $("mon-response").value.trim();
    if (!query || !response) { toast("The question and the answer are both required", "err"); return; }

    busy(btn, true, "Reporting\u2026");
    try {
      const out = await API.record(
        query,
        $("mon-context").value.trim(),
        response,
        $("mon-hallucination").checked
      );
      const m = out.metrics;
      $("record-status").textContent =
        `Reported. So far: ${m.total_predictions} answers checked, ${pct(m.fdr)} false alarms.`;
      $("record-status").className = "status-line inline ok";
      pollStatus();
    } catch (err) {
      $("record-status").textContent = `Couldn't report it: ${err.message}`;
      $("record-status").className = "status-line inline err";
    } finally {
      busy(btn, false);
    }
  });

  // ---------- topics ----------

  let activeDomainCalibrate = null;

  function domainCardHTML(d) {
    const badge = d.drift.drifted
      ? `<span class="badge drifted">needs re-teaching</span>`
      : `<span class="badge ${d.alpha <= 0.02 ? "strict" : "ok"}">${strictness(d.alpha)}</span>`;
    return `
      <div class="domain-card" data-domain="${d.name}">
        <div class="domain-card-head">
          <span class="domain-card-name">${d.name}</span>
          ${badge}
        </div>
        <div class="domain-card-body">
          <span class="k">tolerance:</span> ${pct(d.alpha)} (${strictness(d.alpha).toLowerCase()})<br />
          <span class="k">the line:</span> ${d.calibrated ? `${pct(d.threshold)} support` : "not taught yet"}<br />
          <span class="k">answers checked:</span> ${d.metrics.total_predictions}<br />
          <span class="k">false alarms:</span> ${pct(d.metrics.fdr)}<br />
          <span class="k">advice:</span> ${d.drift.recommendation === "Re-calibrate" ? "its data has changed \u2014 teach it again" : "holding its promise"}
        </div>
        <div class="domain-card-actions">
          <button class="btn" data-action="calibrate">Teach it</button>
          <button class="btn" data-action="drift">Check health</button>
        </div>
      </div>`;
  }

  async function refreshDomains() {
    try {
      const { domains } = await API.listDomains();
      const container = $("domain-cards");
      if (!domains.length) {
        container.innerHTML = `<p class="panel-hint" style="margin:0">No topics yet. Add one above \u2014 try \u201cmedicine\u201d with a tolerance of 0.001 (one mistake per thousand answers).</p>`;
        return;
      }
      container.innerHTML = domains.map(domainCardHTML).join("");
    } catch (err) {
      toast(`Couldn't load topics: ${err.message}`, "err");
    }
  }

  $("register-domain-btn").addEventListener("click", async () => {
    const name = $("domain-name").value.trim();
    const alpha = parseFloat($("domain-alpha").value);
    if (!name) { toast("Give the topic a name", "err"); return; }
    try {
      await API.registerDomain(name, alpha);
      $("domain-register-status").textContent =
        `Added "${name}" \u2014 ${strictness(alpha).toLowerCase()} (${pct(alpha)} tolerance).`;
      $("domain-register-status").className = "status-line ok";
      $("domain-name").value = "";
      refreshDomains();
    } catch (err) {
      $("domain-register-status").textContent = `Couldn't add it: ${err.message}`;
      $("domain-register-status").className = "status-line err";
    }
  });

  $("domain-cards").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const name = btn.closest(".domain-card").dataset.domain;

    if (btn.dataset.action === "calibrate") {
      activeDomainCalibrate = name;
      $("domain-calibrate-name").textContent = name;
      $("domain-calibrate-panel").classList.remove("hidden");
      $("domain-calibrate-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else if (btn.dataset.action === "drift") {
      try {
        await API.domainDrift(name);
        refreshDomains();
        toast(`Checked "${name}"`, "ok");
      } catch (err) {
        toast(`Health check failed: ${err.message}`, "err");
      }
    }
  });

  $("domain-calibrate-btn").addEventListener("click", async () => {
    const btn = $("domain-calibrate-btn");
    if (!activeDomainCalibrate) return;
    const rows = parseLines("domain-cal-rows").map((line) => {
      const parts = line.split("|").map((p) => p.trim());
      if (parts.length !== 4) return null;
      const label = parseInt(parts[3], 10);
      if (label !== 0 && label !== 1) return null;
      return { query: parts[0], context: parts[1], response: parts[2], label };
    });
    if (!rows.length || rows.some((r) => r === null)) {
      $("domain-calibrate-status").textContent =
        "Each line needs four parts separated by | \u2014 question | sources | answer | 0 or 1.";
      $("domain-calibrate-status").className = "status-line inline err";
      return;
    }
    busy(btn, true, "Teaching\u2026");
    try {
      const stats = await API.calibrateDomain(activeDomainCalibrate, rows);
      $("domain-calibrate-status").textContent =
        `Taught "${activeDomainCalibrate}" on ${stats.n_samples} examples \u2014 the line is now at ${pct(stats.threshold)} support.`;
      $("domain-calibrate-status").className = "status-line inline ok";
      refreshDomains();
    } catch (err) {
      $("domain-calibrate-status").textContent = `Couldn't teach it: ${err.message}`;
      $("domain-calibrate-status").className = "status-line inline err";
    } finally {
      busy(btn, false);
    }
  });

  viewHooks.topics = refreshDomains;

  // ---------- boot ----------

  refreshEngineStatus();
})();
