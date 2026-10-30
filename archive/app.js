/* Building damage grade prediction · front end (vanilla JS, no external libraries)
 * Fills the sections of index.html from the JSON API and wires up the predictor and area view.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 1) => (Number(x) * 100).toFixed(d) + "%";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const GRADE_CLASS = { 1: "g1", 2: "g2", 3: "g3" };
  const GRADE_NAME = { 1: "Grade 1 · low", 2: "Grade 2 · medium", 3: "Grade 3 · near-complete" };
  const PROVINCE_COLOURS = ["#cfe0f1", "#d7ecd9", "#f6e7c6", "#e4d7ef", "#f6d6cf", "#cdeee9", "#e8e6da"];

  // Form groups. Any field not matched here is placed in "Other" so every field is shown.
  const FORM_GROUPS = [
    { title: "Location (anonymised geographic IDs)", match: (n) => n.startsWith("geo_level_") },
    { title: "Size and age", names: ["count_floors_pre_eq", "age", "area_percentage", "height_percentage"] },
    { title: "Site, ownership and construction",
      names: ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
              "other_floor_type", "position", "plan_configuration", "legal_ownership_status", "count_families"] },
    { title: "Superstructure (tick all that apply)", match: (n) => n.startsWith("has_superstructure_"), checks: true },
    { title: "Secondary use (tick all that apply)", match: (n) => n.startsWith("has_secondary_use"), checks: true },
  ];

  let OPTIONS = null;
  let RESULT_PLACEHOLDER = "";

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch (e) { body = null; }
    if (!r.ok) {
      const err = new Error("HTTP " + r.status);
      err.status = r.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  function showError(el, msg) {
    el.innerHTML = '<p class="note-box">Could not load this section. ' + esc(msg) + "</p>";
  }

  // ------------------------------------------------------------------ hero and results
  function renderHero(res) {
    const final = res.models.find((m) => m.kind === "ensemble");
    $("hero-acc").textContent = fmt(final.accuracy, 2);
    $("hero-f1").textContent = final.macro_f1.toFixed(3);
  }

  function renderDist(res) {
    const dist = res.class_distribution;
    $("dist").innerHTML =
      ["1", "2", "3"].map((k) =>
        '<div class="row"><span>' + esc(GRADE_NAME[k]) + '</span>' +
        '<div class="track"><div class="fill ' + GRADE_CLASS[k] + '" style="width:' + (dist[k] * 100).toFixed(2) + '%"></div></div>' +
        '<span class="chart-val">' + fmt(dist[k], 2) + "</span></div>").join("") +
      '<p class="note">Share of buildings in each damage grade. Grade 1 is the rarest class.</p>';
  }

  function metricValue(m, key) {
    if (key.indexOf("f1.") === 0) return m.f1[key.slice(3)];
    return m[key];
  }

  function renderResultsTable(res) {
    const models = res.models;
    const cols = [
      { key: "accuracy", label: "Accuracy", fmt: (v) => fmt(v, 2) },
      { key: "macro_f1", label: "Macro-F1", fmt: (v) => v.toFixed(3) },
      { key: "weighted_f1", label: "Weighted F1", fmt: (v) => v.toFixed(3) },
      { key: "balanced_accuracy", label: "Balanced acc.", fmt: (v) => fmt(v, 2) },
      { key: "f1.1", label: "F1 grade 1", fmt: (v) => v.toFixed(3) },
      { key: "f1.2", label: "F1 grade 2", fmt: (v) => v.toFixed(3) },
      { key: "f1.3", label: "F1 grade 3", fmt: (v) => v.toFixed(3) },
    ];
    const kindLabel = { classical: "Classical", boosting: "Boosting", ensemble: "Ensemble" };
    // best value per column among the models reproduced here (the literature row is excluded)
    const best = {};
    cols.forEach((c) => {
      best[c.key] = Math.max(...models.map((m) => metricValue(m, c.key)));
    });

    const head = "<thead><tr><th>Model</th><th>Type</th>" +
      cols.map((c) => '<th class="num-col">' + c.label + "</th>").join("") + "</tr></thead>";
    const body = models.map((m) => {
      const isFinal = m.kind === "ensemble";
      const cells = cols.map((c) => {
        const v = metricValue(m, c.key);
        const txt = c.fmt(v);
        const isBest = Math.abs(v - best[c.key]) < 1e-12;
        return '<td class="num-col">' + (isBest ? "<b>" + txt + "</b>" : txt) + "</td>";
      }).join("");
      return '<tr class="' + (isFinal ? "final" : "") + '"><th>' + esc(m.name) +
        (isFinal ? '<span class="tag gold">shipped</span>' : "") + "</th><td>" +
        esc(kindLabel[m.kind] || m.kind) + "</td>" + cells + "</tr>";
    }).join("");
    const ref = res.reference;
    const refRow = '<tr class="ref"><th>' + esc(ref.name) + '<span class="tag">literature</span></th><td>' +
      "SMOTE-XGBoost</td>" + cols.map((c) => {
        const v = c.key === "macro_f1" ? ref.macro_f1 : null;
        return '<td class="num-col">' + (v == null ? "—" : v.toFixed(3)) + "</td>";
      }).join("") + "</tr>";
    $("results-table").innerHTML = head + "<tbody>" + body + refRow + "</tbody>";
    $("results-note").innerHTML =
      "Test split: " + res.rows_test.toLocaleString() + " buildings (20%, stratified, seed 42). " +
      "Accuracy equals micro-F1 for single-label classification. <b>Bold</b> marks the best model reproduced here. " +
      "The P4 row is the base paper's reported macro-F1 on its own split and preprocessing, so it is <i>not directly comparable</i> to these rows.";
  }

  function renderAccChart(res) {
    const lo = 0.6, hi = 0.8;
    const rows = res.models.map((m) => {
      const w = Math.max(0, Math.min(1, (m.accuracy - lo) / (hi - lo))) * 100;
      return '<div class="chart-row"><span class="lbl">' + esc(m.name) + '</span>' +
        '<div class="track"><div class="fill ' + (m.kind === "ensemble" ? "final" : "") + '" style="width:' + w.toFixed(2) + '%"></div></div>' +
        '<span class="val">' + fmt(m.accuracy, 2) + "</span></div>";
    }).join("");
    $("acc-chart").innerHTML = rows +
      '<div class="axis" style="margin-right:82px"><span>60%</span><span>70%</span><span>80%</span></div>' +
      '<p class="note">Axis starts at 60% so that small differences are visible.</p>';
  }

  function renderF1Chart(res) {
    const base = res.models[0].f1;
    const fin = res.models.find((m) => m.kind === "ensemble").f1;
    const groups = ["1", "2", "3"].map((k) =>
      '<div class="f1-group"><div class="name">' + esc(GRADE_NAME[k]) + "</div>" +
      '<div class="f1-bar"><span>Baseline NB</span><div class="track"><div class="fill" style="width:' +
      (base[k] * 100).toFixed(1) + '%;background:#9aa7b6"></div></div><span class="chart-val">' + base[k].toFixed(3) + "</span></div>" +
      '<div class="f1-bar"><span>Final model</span><div class="track"><div class="fill ' + GRADE_CLASS[k] + '" style="width:' +
      (fin[k] * 100).toFixed(1) + '%"></div></div><span class="chart-val">' + fin[k].toFixed(3) + "</span></div></div>").join("");
    $("f1-chart").innerHTML = groups + '<p class="note">F1 on the 20% test split. Higher is better.</p>';
  }

  function renderCM(res) {
    const cm = res.final_confusion_matrix;
    const names = res.class_names;
    let html = "<div></div>" + ["1", "2", "3"].map((k) =>
      '<div class="h">Predicted<br>' + esc(names[k]) + "</div>").join("");
    for (let i = 0; i < 3; i++) {
      const rowTotal = cm[i].reduce((a, b) => a + b, 0);
      html += '<div class="rl">Actual ' + esc(names[String(i + 1)]) + "</div>";
      for (let j = 0; j < 3; j++) {
        const p = cm[i][j] / rowTotal;
        const alpha = 0.06 + 0.9 * p;
        const color = p > 0.5 ? "#fff" : "#1c2530";
        html += '<div class="c" style="background:rgba(20,33,61,' + alpha.toFixed(3) + ");color:" + color + '">' +
          cm[i][j].toLocaleString() + "<small>" + fmt(p, 1) + " of row</small></div>";
      }
    }
    const total = cm.flat().reduce((a, b) => a + b, 0);
    const correct = cm[0][0] + cm[1][1] + cm[2][2];
    $("cm").innerHTML = '<div class="cm">' + html + "</div>" +
      '<p class="cm-legend">Counts on the test split (' + total.toLocaleString() + " buildings). Shade shows the share of each actual grade's row. " +
      "Correct: " + correct.toLocaleString() + " (" + fmt(correct / total, 2) + "). Most errors are grade 3 buildings predicted as grade 2.</p>";
  }

  // ------------------------------------------------------------------ map
  function collectCoords(geom, visit) {
    const walk = (c) => (typeof c[0] === "number" ? visit(c) : c.forEach(walk));
    walk(geom.coordinates);
  }

  function ringsPath(rings, project) {
    return rings.map((ring) =>
      ring.map((pt, i) => {
        const [x, y] = project(pt);
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ") + "Z").join(" ");
  }

  function renderMap(gj) {
    const feats = gj.features;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    feats.forEach((f) => collectCoords(f.geometry, ([x, y]) => {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }));
    const kx = Math.cos(((minY + maxY) / 2) * Math.PI / 180);
    const W = 560, scale = (W - 10) / ((maxX - minX) * kx);
    const H = (maxY - minY) * scale + 10;
    const project = ([lon, lat]) => [5 + (lon - minX) * kx * scale, 5 + (maxY - lat) * scale];

    const provinces = Array.from(new Set(feats.map((f) => f.properties.province))).sort();
    const colourOf = {};
    provinces.forEach((p, i) => { colourOf[p] = PROVINCE_COLOURS[i % PROVINCE_COLOURS.length]; });

    const paths = feats.map((f) => {
      const g = f.geometry;
      const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
      const d = polys.map((p) => ringsPath(p, project)).join(" ");
      const p = f.properties;
      return '<path class="dist" d="' + d + '" style="--c:' + colourOf[p.province] + '" data-name="' +
        esc(p.name) + '" data-province="' + esc(p.province) + '" tabindex="0" aria-label="' + esc(p.name) +
        ' district"><title>' + esc(p.name) + " district (" + esc(p.province) + " Province)</title></path>";
    }).join("");

    const legend = '<div class="legend">' + provinces.map((p) =>
      '<span><i style="background:' + colourOf[p] + '"></i>' + esc(p) + " Province</span>").join("") + "</div>";

    $("map").innerHTML = '<svg class="svg-map" viewBox="0 0 ' + W + " " + H.toFixed(0) +
      '" role="img" aria-label="Map of Nepal district boundaries">' + paths + "</svg>" + legend;

    const tip = $("map-tip");
    const show = (el) => {
      tip.textContent = el.dataset.name + " district · " + el.dataset.province + " Province";
    };
    $("map").addEventListener("mouseover", (e) => { if (e.target.classList.contains("dist")) show(e.target); });
    $("map").addEventListener("focusin", (e) => { if (e.target.classList.contains("dist")) show(e.target); });
    $("map").addEventListener("mouseleave", () => { tip.textContent = "Hover a district to see its name."; });
  }

  // ------------------------------------------------------------------ area table
  function renderAreas(data) {
    const o = data.overall;
    $("overall").innerHTML =
      '<div class="stat-s"><b>' + fmt(o.accuracy, 1) + "</b><span>accuracy (test)</span></div>" +
      '<div class="stat-s"><b>' + o.buildings_test.toLocaleString() + "</b><span>test buildings</span></div>" +
      '<div class="stat-s"><b>' + fmt(o.actual_share["3"], 1) + "</b><span>actual grade 3</span></div>";

    const rows = data.areas.slice().sort((a, b) => b.risk_score - a.risk_score);
    const tbody = document.querySelector("#area-table tbody");
    tbody.innerHTML = rows.map((a) =>
      '<tr class="clickable" tabindex="0" data-id="' + a.geo_level_1_id + '">' +
      "<td>Area " + a.geo_level_1_id + "</td>" +
      '<td class="num-col">' + a.buildings_test.toLocaleString() + "</td>" +
      '<td class="num-col">' + fmt(a.actual_share["3"], 1) + "</td>" +
      '<td class="num-col"><span class="mini"><span class="mini-fill" style="width:' + (a.risk_score * 100).toFixed(1) +
      '%"></span></span>' + a.risk_score.toFixed(3) + "</td>" +
      '<td class="num-col">' + fmt(a.accuracy, 1) + "</td></tr>").join("");

    const byId = {};
    rows.forEach((a) => { byId[a.geo_level_1_id] = a; });
    const select = (id) => {
      const a = byId[id];
      if (!a) return;
      tbody.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === String(id)));
      $("area-detail").innerHTML =
        '<h4>Area ' + a.geo_level_1_id + " · " + a.buildings_test.toLocaleString() + " buildings</h4>" +
        '<div class="table-wrap"><table class="data"><thead><tr><th>Grade</th><th class="num-col">Actual share</th>' +
        '<th class="num-col">Predicted share</th><th class="num-col">Mean probability</th></tr></thead><tbody>' +
        ["1", "2", "3"].map((k) =>
          "<tr><th>" + esc(GRADE_NAME[k]) + '</th><td class="num-col">' + fmt(a.actual_share[k], 1) +
          '</td><td class="num-col">' + fmt(a.predicted_share[k], 1) + '</td><td class="num-col">' +
          fmt(a.mean_probability[k], 1) + "</td></tr>").join("") +
        '</tbody></table></div><p class="note">Predicted share: share of buildings whose most likely grade is k. ' +
        "Accuracy in this area: " + fmt(a.accuracy, 1) + ".</p>";
    };
    tbody.addEventListener("click", (e) => {
      const tr = e.target.closest("tr");
      if (tr) select(tr.dataset.id);
    });
    tbody.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && e.target.matches("tr")) {
        e.preventDefault();
        select(e.target.dataset.id);
      }
    });
    if (rows.length) select(rows[0].geo_level_1_id);
  }

  // ------------------------------------------------------------------ predictor form
  function fieldHTML(f) {
    const range = (f.min != null && f.max != null)
      ? ' <em class="range">(' + f.min + "–" + f.max + ")</em>" : "";
    if (f.kind === "choice") {
      return '<label class="field"><span>' + esc(f.label) + '</span><select name="' + f.name + '">' +
        f.values.map((v) => '<option value="' + esc(v) + '"' + (String(v) === String(f.default) ? " selected" : "") +
          ">" + esc(v) + "</option>").join("") + "</select></label>";
    }
    if (f.kind === "number" || f.kind === "geo") {
      const step = f.kind === "geo" ? "1" : "any";
      return '<label class="field"><span>' + esc(f.label) + range + '</span><input type="number" name="' + f.name +
        '" min="' + f.min + '" max="' + f.max + '" step="' + step + '" value="' + f.default + '" required></label>';
    }
    return "";
  }

  function checkHTML(f) {
    return '<label class="check"><input type="checkbox" name="' + f.name + '" value="1"' +
      (Number(f.default) === 1 ? " checked" : "") + "> " + esc(f.label) + "</label>";
  }

  function buildForm() {
    const fields = OPTIONS.fields;
    const used = new Set();
    const groups = FORM_GROUPS.map((g) => {
      const members = fields.filter((f) => {
        const hit = (g.names && g.names.indexOf(f.name) >= 0) || (g.match && g.match(f.name));
        if (hit) used.add(f.name);
        return hit;
      });
      return { ...g, members };
    });
    const other = fields.filter((f) => !used.has(f.name));
    if (other.length) groups.push({ title: "Other", members: other });

    $("form-fields").innerHTML = groups.filter((g) => g.members.length).map((g) => {
      const inner = g.checks
        ? '<div class="checks">' + g.members.map((f) => (f.kind === "binary" ? checkHTML(f) : fieldHTML(f))).join("") + "</div>"
        : '<div class="fields">' + g.members.map((f) => (f.kind === "binary" ? checkHTML(f) : fieldHTML(f))).join("") + "</div>";
      return '<fieldset class="group"><legend>' + esc(g.title) + "</legend>" + inner + "</fieldset>";
    }).join("");
  }

  function setError(msg) {
    const el = $("form-error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  function collectPayload(form) {
    const payload = {};
    const problems = [];
    OPTIONS.fields.forEach((f) => {
      const el = form.elements.namedItem(f.name);
      if (!el) { problems.push(f.label + " is missing from the form"); return; }
      if (f.kind === "binary") {
        payload[f.name] = el.checked ? 1 : 0;
      } else if (f.kind === "number" || f.kind === "geo") {
        const v = f.kind === "geo" ? parseInt(el.value, 10) : parseFloat(el.value);
        if (Number.isNaN(v)) problems.push("Enter a number for " + f.label + ".");
        else if (v < f.min || v > f.max) problems.push(f.label + " must be between " + f.min + " and " + f.max + ".");
        payload[f.name] = v;
      } else {
        payload[f.name] = el.value;
      }
    });
    if (problems.length) throw new Error(problems.join(" "));
    return payload;
  }

  function describeError(err) {
    const d = err && err.body && err.body.detail;
    if (d && typeof d === "object") {
      const parts = [];
      if (d.missing) parts.push("Missing: " + d.missing.join(", ") + ".");
      if (d.invalid) parts.push(Object.entries(d.invalid).map(([k, v]) => k + ": " + v).join("; ") + ".");
      if (parts.length) return parts.join(" ");
    }
    if (typeof d === "string") return d;
    return "Prediction failed (" + (err && err.message ? err.message : "unknown error") + ").";
  }

  function renderPrediction(out) {
    const g = String(out.damage_grade);
    const p = out.probabilities;
    $("result").innerHTML =
      "<h3>Result</h3>" +
      '<div class="grade ' + GRADE_CLASS[g] + '"><div class="lbl">Predicted damage grade ' + esc(g) + "</div>" +
      '<div class="big">' + esc(out.label) + "</div></div>" +
      '<p class="guide">' + esc(out.guidance) + "</p>" +
      '<p class="sub">Class probabilities</p>' +
      ["1", "2", "3"].map((k) =>
        '<div class="bar-row"><span>Grade ' + k + '</span><div class="bar"><div class="bar-fill ' + GRADE_CLASS[k] +
        '" style="width:' + (p[k] * 100).toFixed(1) + '%"></div></div><span class="bar-val">' + fmt(p[k], 1) + "</span></div>").join("") +
      '<p class="meta">Model: ' + esc(out.model) + "</p>" +
      '<p class="disclaimer">' + esc(out.disclaimer) + "</p>";
  }

  function initForm() {
    const form = $("predict-form");
    RESULT_PLACEHOLDER = $("result").innerHTML;
    buildForm();
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      setError("");
      const btn = $("submit-btn");
      let payload;
      try { payload = collectPayload(form); } catch (err) { setError(err.message); return; }
      btn.disabled = true;
      btn.textContent = "Predicting…";
      try {
        const out = await getJSON("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        renderPrediction(out);
      } catch (err) {
        setError(describeError(err));
      } finally {
        btn.disabled = false;
        btn.textContent = "Predict damage grade";
      }
    });
    $("reset-form").addEventListener("click", () => {
      buildForm();
      setError("");
      $("result").innerHTML = RESULT_PLACEHOLDER;
    });
  }

  // ------------------------------------------------------------------ boot
  async function section(fn) {
    try { await fn(); } catch (e) { console.error(e); }
  }

  async function init() {
    // Each block loads on its own, so one failing endpoint does not blank the page.
    const results = await getJSON("/api/results").catch((e) => { console.error(e); return null; });
    if (results) {
      await section(() => renderHero(results));
      await section(() => renderDist(results));
      await section(() => renderResultsTable(results));
      await section(() => renderAccChart(results));
      await section(() => renderF1Chart(results));
      await section(() => renderCM(results));
    } else {
      ["dist", "results-table", "acc-chart", "f1-chart", "cm"].forEach((id) => showError($(id), "results.json is unavailable."));
    }

    await section(async () => {
      renderMap(await getJSON("/api/districts"));
    });
    await section(async () => {
      renderAreas(await getJSON("/api/areas"));
    });

    try {
      OPTIONS = await getJSON("/api/options");
      initForm();
    } catch (e) {
      console.error(e);
      $("form-fields").innerHTML = '<p class="note-box">The predictor is unavailable (' + esc(e.message) + ").</p>";
      $("submit-btn").disabled = true;
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
