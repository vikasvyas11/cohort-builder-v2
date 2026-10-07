// Step 4: the results of one run, in seven tabs. Run 1 and Run 2 both render through renderRun(),
// so they always show the same analysis.

import { api } from "./api.js";
import { S, go } from "./state.js";
import { debounce, download, fmt, h, livePlot, metrics, note, num4, plot, plots, put, switchEl, table, tabs, fill } from "./ui.js";

// ── tabs ──────────────────────────────────────────────────────────────────────

function edgesTab(r) {
  const e = r.edges;
  return h("div", {},
    e.prob_stats.length ? h("div", {}, h("h3", {}, "Match probability statistics"), table(e.prob_stats)) : null,
    e.weight_percentiles.length ? h("div", {}, h("h3", {}, "Match weight percentiles"), table(e.weight_percentiles)) : null,
    ...e.charts.map((c) => h("div", {}, plot(c.figure), h("p", { class: "muted" }, c.caption))));
}

function clustersTab(r) {
  const c = r.clusters;
  return h("div", {},
    metrics([["Total clusters", fmt(c.n_clusters)], ["Cross-dataset clusters", fmt(c.n_cross_dataset)]], "two"),
    c.singleton.length ? h("div", { class: "grid two" },
      h("div", {}, h("h3", {}, "Single-record vs multi-record clusters"), table(c.singleton),
        h("p", { class: "muted" }, "Many single-record clusters mean many records could not be linked; multi-record clusters are the duplicates or cross-dataset matches found.")),
      plot(c.singleton_figure, 280)) : null,
    c.stats.length ? table(c.stats) : null,
    c.size_figure ? h("div", {}, plot(c.size_figure), h("p", { class: "muted" }, "Many size-1 clusters and a few large ones is typical. Very large clusters may mean over-linking.")) : null,
    c.venn ? h("div", {}, h("h3", {}, "Clusters by source dataset"), table(c.venn.rows), plot(c.venn.figure)) : null);
}

function demographicsTab(slot) {
  const box = h("div", {}, h("p", { class: "muted", role: "status" }, "Loading..."));
  api.demographics(S.sid, slot).then((d) => {
    const l = d.linked_vs_unlinked;
    fill(box, 
      d.legacy.length ? plots(d.legacy) : null,
      d.breakdowns.length ? plots(d.breakdowns.map((b) => b.figure)) : null,
      !d.legacy.length && !d.breakdowns.length ? h("p", { class: "muted" }, "No recognised demographic columns in this data.") : null,
      h("h3", {}, "Linked vs unlinked"),
      l.fields.length ? h("div", {},
        ...l.fields.map((f) => h("div", { class: "grid two" }, plot(f.linked, 280), plot(f.unlinked, 280))),
        h("ul", {}, l.summary.map((s) => h("li", {}, s))),
        h("p", { class: "muted" }, `Linked: ${fmt(l.n_linked)} records in clusters of more than one. Unlinked: ${fmt(l.n_unlinked)} in single-record clusters.`))
        : h("p", { class: "muted" }, "No comparable demographic fields."));
  }).catch((e) => fill(box, note("bad", e.message)));
  return box;
}

function explorerTab(slot, r) {
  const toggles = Object.fromEntries(r.rules.filter((x) => x.on).map((x) => [x.rule, true]));
  let threshold = 0.8;
  const rulesBox = h("div", { class: "grid three" });
  const live = h("div", {}, h("p", { class: "muted", role: "status" }, "Loading..."));
  const tail = h("div", {});
  const wfLeft = livePlot(), wfRight = livePlot();     // redrawn in place on every toggle
  let counts = {};

  // Built once, so keyboard focus stays on the switch you just used; only the text and look update.
  const cards = Object.keys(toggles).map((k) => {
    const info = h("span", { class: "muted" });
    const card = h("div", { class: "rule" },
      switchEl(toggles[k], (on) => { toggles[k] = on; drawRules(); refresh(); }, `Use ${k}`),
      h("div", {}, h("strong", {}, k), h("code", {}, k.split("+").map((p) => `l."${p}" = r."${p}"`).join(" AND ")), info));
    return { k, info, card };
  });
  fill(rulesBox, ...cards.map((c) => c.card));
  const drawRules = () => cards.forEach(({ k, info, card }) => {
    info.textContent = `${toggles[k] ? "On" : "Off"}, ${fmt(counts[k] || 0)} pairs`;
    card.classList.toggle("off", !toggles[k]);
  });

  const refresh = debounce(async () => {
    try {
      const d = await api.explorer(S.sid, slot, { toggles, threshold });
      counts = Object.fromEntries(d.rules.map((x) => [x.rule, x.pairs]));
      drawRules();
      const s = d.stats;
      fill(live, 
        d.waterfall && d.waterfall.fields.length ? h("div", {}, h("h3", {}, "Blocking rule cascade"),
          h("p", { class: "muted" }, "Switching a rule off hands its pairs to later rules that also cover them, or loses them."),
          h("div", { class: "grid two" }, h("div", {}, h("strong", {}, d.waterfall.titles[0]), wfLeft),
            h("div", {}, h("strong", {}, d.waterfall.titles[1]), wfRight))) :
          d.and_mode ? note("", "This run used AND blocking: all fields form one rule, so there is no cascade between rules.") : null,
        metrics([["Candidate pairs", fmt(s.candidate_pairs)], ["Rules on", `${s.rules_on}/${s.rules_total}`],
          ["Reduction", `${s.reduction_pct}%`], ["Original pairs", fmt(s.original_pairs)]]),
        h("h3", {}, "Pairs (first 200)"), table(d.table, null, { max: 200 }),
        ...d.match_quality.map((m) => h("div", {}, h("h3", {}, `Match quality by ${m.field}`), table(m.rows))));
      if (d.waterfall && d.waterfall.fields.length) { wfLeft.set(d.waterfall.left); wfRight.set(d.waterfall.right); }
    } catch (error) {
      fill(live, note("bad", error.message));
    }
  }, 200);

  const slider = h("input", { type: "range", min: 0.5, max: 0.99, step: 0.01, value: 0.8, "aria-label": "Cluster threshold for the explorer" });
  const out = h("strong", {}, "0.80");
  slider.oninput = () => { threshold = Number(slider.value); out.textContent = threshold.toFixed(2); };
  const result = h("div", {});
  const button = h("button", { class: "primary" }, "Re-cluster with active rules");
  button.onclick = async () => {
    button.disabled = true;
    try {
      const c = await api.recluster(S.sid, slot, { toggles, threshold });
      fill(result, note("ok", `Re-clustered: ${fmt(c.new_clusters)} clusters from ${fmt(c.edges)} edges (original rules: ${fmt(c.original_clusters)}).`));
    } catch (error) {
      fill(result, note("bad", error.message));
    } finally { button.disabled = false; }
  };
  tail.append(h("div", { class: "row" }, h("label", {}, "Cluster threshold", slider), out, button), result);
  drawRules();
  refresh();
  return h("div", {}, h("p", {}, "Switch blocking rules on or off to see which candidate pairs remain, and what it does to the clusters."),
    h("h3", {}, "Rules"), rulesBox, live, tail);
}

function studioTab(slot) {
  const frame = h("iframe", { class: "studio", title: "Cluster studio", sandbox: "allow-scripts" });
  api.studio(S.sid, slot).then((html) => { frame.srcdoc = html; }).catch(() => { frame.srcdoc = "<p>Cluster studio is not available.</p>"; });
  return h("div", {}, h("p", { class: "muted" }, "Each node is a record and each edge a predicted match. Use it to inspect clusters by eye."), frame);
}

function accuracyTab(r) {
  const a = r.accuracy;
  if (!a.available) {
    return h("div", {}, note("", a.reason), h("p", { class: "muted" }, "Accuracy needs a ground-truth column named 'cluster'. The built-in datasets and generated Dataset B have one."));
  }
  const cm = a.cm;
  return h("div", {},
    h("p", { class: "muted" }, `Pairs count as predicted at match probability of at least ${(cm.min_match_probability || 0).toFixed(2)} (this run's cluster threshold). The curve below covers every threshold.`),
    metrics([["True positives", fmt(cm.tp)], ["False positives", fmt(cm.fp)], ["False negatives", fmt(cm.fn)], ["True pairs", fmt(cm.n_gt_edges)]]),
    h("div", { class: "grid two" }, h("div", {}, h("h3", {}, "Derived metrics"),
      table(a.derived.map((d) => ({ metric: d.metric, value: num4(d.value), meaning: d.meaning })))), plot(a.figure, 280)),
    a.curve_figures.length ? h("div", {}, h("h3", {}, "Precision-recall curve and threshold summary"), plots(a.curve_figures),
      a.curve_summary.average_precision != null ? metrics([
        ["Average precision", num4(a.curve_summary.average_precision)], ["Best F1", num4(a.curve_summary.best_f1)],
        ["Threshold for best F1", Number(a.curve_summary.best_f1_threshold).toFixed(3)], ["Best F*", num4(a.curve_summary.best_fstar)]]) : null) : null);
}

function rawTab(slot) {
  const box = h("div", {}, h("p", { class: "muted", role: "status" }, "Loading..."));
  api.raw(S.sid, slot).then((d) => fill(box, 
    h("h3", {}, `Candidate pairs (first 100 of ${fmt(d.n_predict)})`), table(d.predict, null, { max: 100 }),
    h("p", { class: "muted" }, "Each row is a candidate pair. gamma_ columns show field agreement (1 exact, 0 disagree); match_key names the blocking rule that generated it."),
    h("h3", {}, `Clusters (first 100 of ${fmt(d.n_cluster)})`), table(d.cluster, null, { max: 100 })),
  ).catch((e) => fill(box, note("bad", e.message)));
  return box;
}

// ── a whole run ───────────────────────────────────────────────────────────────

export async function renderRun(slot) {
  const r = await api.run(S.sid, slot);
  const c = r.cards;
  return h("div", {},
    r.zero_edges ? note("bad", "This run produced 0 candidate pairs, so every record is in its own cluster. The table shows why, field by field.") : null,
    r.zero_edges ? table(r.zero_edge_diagnostic) : null,
    h("h2", {}, "Summary"),
    metrics([["Records processed", fmt(c.records)], ["Predicted edges", fmt(c.edges)], ["Distinct entity clusters", fmt(c.clusters)],
      ["Unique IDs with a match", fmt(c.unique_ids)]]),
    tabs([["Edge Metrics", () => edgesTab(r)], ["Cluster Metrics", () => clustersTab(r)],
      ["Demographics", () => demographicsTab(slot)], ["Blocking Explorer", () => explorerTab(slot, r)],
      ["Cluster Studio", () => studioTab(slot)], ["Confusion Matrix", () => accuracyTab(r)], ["Raw Data", () => rawTab(slot)]]));
}

// ── shared blocks ─────────────────────────────────────────────────────────────

export function assistantBlock() {
  return h("section", { class: "card" },
    h("h2", {}, "Need help reading these results?"),
    h("p", { class: "muted" }, "The Cohort Builder AI Assistant is trained on Splink, record-linkage theory and this app's documentation. Use it to interpret the results and decide which parameters to adjust. It opens in a new tab and does not receive your data."),
    S.assistantUrl ? h("a", { class: "btn primary", href: S.assistantUrl, target: "_blank", rel: "noopener" }, "Open the AI Assistant") : null);
}

export function reportBlock(slots) {
  return h("section", { class: "card" }, h("h2", {}, "Downloads"),
    h("div", { class: "stack" }, slots.map((slot) => h("div", { class: "row" }, h("strong", {}, slot === "run1" ? "Run 1" : "Run 2"),
      download(api.download(S.sid, slot, "report.html"), "Report (HTML)"),
      download(api.download(S.sid, slot, "model.json"), "Model JSON"),
      download(api.download(S.sid, slot, "cohort.csv"), "Cohort CSV")))),
    h("p", { class: "muted" }, "The report opens offline and prints to PDF. The model JSON can be loaded in the Advanced flow to skip training."));
}

export async function renderResults(view) {
  put(view, h("h1", {}, S.flow === "advanced" ? "Results (Advanced flow)" : "Results"));
  put(view, await renderRun("run1"));
  put(view, assistantBlock(), reportBlock(["run1"]),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => go("compare") }, "Continue to compare runs")));
}
