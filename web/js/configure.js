// Step 3: choose fields, blocking rules, mode and method, then run. The same page serves all three flows;
// in the Advanced flow the model supplies fields and rules, so only the method, mode and threshold show.

import { api, runAndWait } from "./api.js";
import { S, go, save } from "./state.js";
import { chip, debounce, fmt, h, note, opts, put, runStages, stages, fill } from "./ui.js";

const META = new Set(["unique_id", "cluster", "source_dataset"]);

function datasetBPanel(rerender) {
  const o = S.overview;
  const status = h("div");
  const eligible = o.column_info.filter((c) => !META.has(c.column) && c.type !== "id").map((c) => c.column);
  const rates = {};
  const frac = h("input", { type: "range", min: 0.1, max: 0.9, step: 0.05, value: 0.5, "aria-label": "Sample fraction" });
  const fracLabel = h("strong", {}, "50%");
  frac.oninput = () => { fracLabel.textContent = `${Math.round(frac.value * 100)}%`; };
  const letters = h("input", { type: "number", min: 1, max: 5, value: 1, "aria-label": "Letters changed per damaged value" });
  const shift = h("input", { type: "number", min: 1, max: 10, value: 1, "aria-label": "Year shift" });
  const rows = eligible.map((f) => {
    const slider = h("input", { type: "range", min: 0, max: 50, value: 0, "aria-label": `Damage rate for ${f}` });
    const out = h("span", { class: "muted" }, "untouched");
    slider.oninput = () => { rates[f] = Number(slider.value) / 100; out.textContent = Number(slider.value) ? `${slider.value}% damaged` : "untouched"; };
    return h("div", { class: "row" }, h("code", {}, f), slider, out);
  });
  const make = h("button", { class: "primary" }, o.rows_b ? "Regenerate Dataset B" : "Generate Dataset B");
  make.onclick = async () => {
    make.disabled = true;
    fill(status, h("span", { class: "muted", role: "status" }, "Generating..."));
    try {
      const result = await api.datasetB(S.sid, { sample_frac: Number(frac.value), rates, letters: Number(letters.value), year_shift: Number(shift.value) });
      S.overview = result;
      S.runs = {};
      save();
      rerender();
    } catch (error) {
      fill(status, note("bad", error.message));
      make.disabled = false;
    }
  };
  return h("details", { class: "card", open: !o.rows_b },
    h("summary", {}, o.rows_b ? `Dataset B: ${fmt(o.rows_b)} records (change or regenerate)` : "Dataset B: none yet. Generate a damaged copy of Dataset A to test linking"),
    h("p", { class: "muted" }, "Dataset B is a sample of Dataset A with typos, shifted dates and missing values added, so every B record has a true match. Choose how much damage each field gets."),
    h("div", { class: "row" }, h("label", {}, "Share of A copied", frac, fracLabel),
      h("label", {}, "Letters changed per value", letters), h("label", {}, "Year shift", shift)),
    h("div", { class: "stack" }, rows), make, status);
}

function fieldPicker(cfg, columns, onchange) {
  const chips = columns.map((c) => chip(c, cfg.fields.includes(c), (on) => {
    cfg.fields = columns.filter((x) => (x === c ? on : cfg.fields.includes(x)));
    if (!on) delete cfg.blocking_toggles[c];
    onchange(true);
  }));
  return h("fieldset", {}, h("legend", {}, "Fields to compare"), h("div", { class: "chips" }, chips));
}

function blockingPanel(cfg, onchange) {
  const singles = cfg.fields.map((f) => chip(f, cfg.blocking_toggles[f], (on) => { cfg.blocking_toggles[f] = on; onchange(); }));
  const combos = Object.keys(cfg.blocking_toggles).filter((k) => k.includes("+"));
  const comboRows = combos.map((k) => h("div", { class: "rule" },
    chip(k, cfg.blocking_toggles[k], (on) => { cfg.blocking_toggles[k] = on; onchange(); }),
    h("code", {}, k.split("+").map((p) => `l."${p}" = r."${p}"`).join(" AND ")),
    h("button", { class: "link", onclick: () => { delete cfg.blocking_toggles[k]; onchange(true); } }, "Remove")));
  const picks = [0, 1, 2].map((i) => h("select", { "aria-label": `Combined rule field ${i + 1}` },
    i === 2 ? h("option", { value: "" }, "(none)") : null, cfg.fields.map((f, j) => h("option", { value: f, selected: i < 2 && j === Math.min(i, cfg.fields.length - 1) }, f))));
  const add = h("button", {}, "Add combined rule");
  add.onclick = () => {
    const parts = [...new Set(picks.map((p) => p.value).filter(Boolean))];
    if (parts.length < 2) return;
    cfg.blocking_toggles[parts.join("+")] = true;
    onchange(true);
  };
  const redundant = [...new Set(combos.filter((k) => cfg.blocking_toggles[k]).flatMap((k) => k.split("+")))].filter((f) => cfg.blocking_toggles[f]);
  return h("fieldset", {}, h("legend", {}, "Blocking rules"),
    h("p", { class: "muted" }, "Two records are compared only if they agree exactly on at least one switched-on rule. Pick selective fields: gender or city alone pair up a large share of the data."),
    h("div", { class: "chips" }, singles),
    h("h3", {}, "Combined rules"),
    h("p", { class: "muted" }, "A pair must agree on every field of a combined rule, for example first_name and last_name together."),
    h("div", { class: "stack" }, comboRows), h("div", { class: "row" }, ...picks, add),
    redundant.length ? note("warn", `${redundant.join(", ")} also has its own single-field rule on. In OR mode that rule already covers everything the combined rule would, so the combined rule adds nothing until you switch it off.`) : null,
    h("h3", {}, "How rules combine"),
    opts("bmode", [["OR", "OR: a pair is a candidate if any rule matches (higher recall)"],
      ["AND", "AND: a pair must agree on every field of every rule (stricter, higher precision)"]], cfg.blocking_mode,
    (v) => { cfg.blocking_mode = v; onchange(); }));
}

function comparisonTypes(cfg) {
  const o = S.overview;
  if (S.flow !== "upload") return null;
  return h("details", { class: "card" }, h("summary", {}, "Comparison type per field"),
    h("div", { class: "stack" }, cfg.fields.map((f) => h("div", { class: "row" }, h("code", {}, f),
      h("select", { "aria-label": `Comparison type for ${f}`, onchange: (e) => { cfg.comp_types[f] = e.target.value; save(); } },
        o.comparison_types.map((t) => h("option", { value: t, selected: (cfg.comp_types[f] || "ExactMatch") === t }, t)))))));
}

function methodPanel(cfg, rerender, onchange) {
  const hp = cfg.hyperparams;
  const num = (key, min, max, step) => h("input", { type: "number", min, max, step, value: hp[key], "aria-label": key, onchange: (e) => { hp[key] = Number(e.target.value); save(); } });
  return h("fieldset", {}, h("legend", {}, "Method"),
    opts("ltype", [["deterministic", "Deterministic: exact-match rules, no training, every match scores 1.0"],
      ["probabilistic", "Probabilistic: a Fellegi-Sunter model trained by EM, which handles typos and gaps"]], cfg.linkage_type,
    (v) => { cfg.linkage_type = v; onchange(); rerender(); }),
    cfg.linkage_type === "probabilistic" && S.flow !== "advanced" ? h("details", {}, h("summary", {}, "Training settings"),
      h("div", { class: "row" }, h("label", {}, "Max EM iterations", num("max_iterations", 5, 500, 5)),
        h("label", {}, "EM convergence", num("em_convergence", 0.00000001, 0.01, 0.0001)),
        h("label", {}, "Recall estimate for the prior", num("recall_estimate", 0.1, 0.99, 0.05)))) : null);
}

export async function renderConfigure(view) {
  const o = S.overview, cfg = S.cfg;
  const advanced = S.flow === "advanced";
  const columns = o.column_info.map((c) => c.column).filter((c) => !META.has(c) && c !== "unique_id" &&
    (S.flow !== "upload" || o.column_info.find((x) => x.column === c).type !== "id"));
  const rerender = () => { fill(view); renderConfigure(view); };
  const status = h("div");
  const estimateBox = h("div", { "aria-live": "polite" });
  const run = h("button", { class: "primary" }, "Run analysis");

  const changed = (structural) => { save(); if (structural) rerender(); else check(); };
  const check = debounce(async () => {
    if (advanced) return;
    try {
      const e = await api.estimate(S.sid, cfg);
      const kind = e.level === "ok" ? "ok" : e.level === "warn" ? "warn" : "bad";
      const text = e.level === "refuse"
        ? `About ${fmt(e.pairs)} candidate pairs: over the ${fmt(e.limit)} this app will score. Switch off low-selectivity rules, use AND, or combine fields.`
        : e.level === "warn" ? `About ${fmt(e.pairs)} candidate pairs. This may be slow; tighten blocking if the run struggles.`
          : `About ${fmt(e.pairs)} candidate pairs. Looks manageable.`;
      const biggest = e.level === "ok" ? "" : ` Biggest rules: ${e.by_rule.slice(0, 3).map((r) => `${r.rule} (${fmt(r.pairs)})`).join(", ")}.`;
      fill(estimateBox, note(kind, text + biggest));
      run.disabled = e.level === "refuse";
    } catch (error) {
      fill(estimateBox, note("warn", error.message));
      run.disabled = false;
    }
  }, 300);

  run.onclick = async () => {
    run.disabled = true;
    const progress = stages(runStages(advanced, cfg.linkage_type));
    fill(status, progress.node);
    try {
      await runAndWait(S.sid, "run1", { ...cfg, from_model: advanced }, (s) => progress.update(s.stage, s.elapsed_s));
      S.runs = { run1: true };
      S.run2Toggles = null;
      save();
      go("results");
    } catch (error) {
      fill(status, note("bad", error.message));
      run.disabled = false;
    }
  };

  put(view, h("h1", {}, advanced ? "Method and run" : "Configure"),
    h("p", { class: "muted" }, advanced ? "The saved model supplies the fields and rules. Choose how to run it." : "Pick what to compare and how to find candidate pairs, then run."));
  put(view, datasetBPanel(rerender));
  put(view, h("fieldset", {}, h("legend", {}, "What to do"),
    opts("op", [["dedupe", "Find duplicates in Dataset A only"],
      ["link_dedupe", `Link Dataset A with Dataset B${o.rows_b ? "" : " (generate Dataset B above first)"}`, !o.rows_b]],
    o.rows_b ? cfg.operation_mode : "dedupe", (v) => { cfg.operation_mode = v; changed(); })));
  if (!o.rows_b) cfg.operation_mode = "dedupe";
  if (!advanced) put(view, fieldPicker(cfg, columns, changed), blockingPanel(cfg, changed), comparisonTypes(cfg));
  put(view, methodPanel(cfg, rerender, changed));
  const slider = h("input", { type: "range", min: 0.5, max: 0.99, step: 0.01, value: cfg.cluster_threshold, "aria-label": "Cluster threshold" });
  const out = h("strong", {}, cfg.cluster_threshold.toFixed(2));
  slider.oninput = () => { cfg.cluster_threshold = Number(slider.value); out.textContent = cfg.cluster_threshold.toFixed(2); save(); };
  put(view, h("fieldset", {}, h("legend", {}, "Cluster threshold"),
    h("p", { class: "muted" }, "Pairs scoring at or above this match probability are grouped into one entity."), h("div", { class: "row" }, slider, out)),
  estimateBox, h("div", { class: "row" }, run), status);
  check();
}
