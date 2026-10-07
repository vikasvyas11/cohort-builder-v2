// Step 5: change the blocking rules and run again, with a live cascading waterfall of the pairs each rule
// contributes. Step 6: export the cohort.

import { api, runAndWait } from "./api.js";
import { S, go, save } from "./state.js";
import { chip, debounce, download, fmt, h, livePlot, metrics, note, num4, plots, put, runStages, stages, table, fill } from "./ui.js";
import { assistantBlock, renderRun, reportBlock } from "./results.js";

const sql = (rule) => rule.split("+").map((p) => `l."${p}" = r."${p}"`).join(" AND ");

// The waterfall: two charts that are drawn once and updated in place, so a toggle feels immediate.
function waterfallPanel() {
  const left = livePlot(), right = livePlot();
  const body = h("div", {}, h("p", { class: "muted", role: "status" }, "Counting pairs for each rule..."));
  const draw = (d) => {
    if (!d.fields || !d.fields.length) {
      fill(body, note("warn", "No rule can be counted on this data yet. Switch on at least one rule that uses a column in the data."));
      return;
    }
    const t = d.totals;
    const skipped = (d.skipped || []).map((s) => note("warn",
      s.pairs == null ? `Rule ${s.rule} uses a column that is not in the data, so it is left out.`
        : `Rule ${s.rule} would add about ${fmt(s.pairs)} pairs, over the ${fmt(s.limit)}-pair limit, so it is left out of the chart.`));
    fill(body, 
      ...skipped,
      metrics([[`${d.titles[0]}: candidate pairs`, fmt(t.before)], [`${d.titles[1]}: candidate pairs`, fmt(t.after), t.change],
        ["Rules on", `${d.rules_on.after}`, d.rules_on.after - d.rules_on.before]]),
      d.and_mode ? note("", "Run 1 used AND blocking, where all fields form one rule. The cascade below treats each rule on its own, as OR blocking would.") : null,
      h("div", { class: "grid two" }, h("div", {}, h("strong", {}, d.titles[0]), left), h("div", {}, h("strong", {}, d.titles[1]), right)),
      table(d.table.map((r) => ({ rule: r.rule, on: r.on ? "on" : "off", [d.titles[0]]: r.before, [d.titles[1]]: r.after, change: r.change }))),
      h("p", { class: "muted" }, "Each pair is credited to the first rule, in the order shown, that covers it, so none is counted twice. Switching a rule off hands its pairs to later rules that also cover them; a combined rule such as first_name+last_name adds pairs only when the single-field rules it contains are off. Red marks what a switched-off rule contributed before."));
    left.set(d.left);
    right.set(d.right);
  };
  return { node: body, draw };
}

function ruleEditor(run1, toggles, onchange) {
  const fields = run1.config.selected_fields;
  const box = h("div", { class: "stack" });
  const draw = () => {
    const keys = Object.keys(toggles);
    fill(box, 
      h("div", { class: "chips" }, keys.filter((k) => !k.includes("+")).map((k) => chip(k, toggles[k], (on) => { toggles[k] = on; onchange(); }))),
      ...keys.filter((k) => k.includes("+")).map((k) => h("div", { class: "rule" },
        chip(k, toggles[k], (on) => { toggles[k] = on; onchange(); }), h("code", {}, sql(k)),
        h("button", { class: "link", onclick: () => { delete toggles[k]; draw(); onchange(); } }, "Remove"))));
  };
  draw();
  const picks = [0, 1, 2].map((i) => h("select", { "aria-label": `Combined rule field ${i + 1}` },
    i === 2 ? h("option", { value: "" }, "(none)") : null, fields.map((f, j) => h("option", { value: f, selected: i < 2 && j === Math.min(i, fields.length - 1) }, f))));
  const add = h("button", {}, "Add combined rule");
  add.onclick = () => {
    const parts = [...new Set(picks.map((p) => p.value).filter(Boolean))];
    if (parts.length < 2) return;
    toggles[parts.join("+")] = true;
    draw();
    onchange();
  };
  return h("div", {}, box, h("h3", {}, "Add a combined rule"),
    h("p", { class: "muted" }, "A pair must agree on every field of the rule. The chart shows what it adds, live."), h("div", { class: "row" }, ...picks, add));
}

export async function renderCompare(view) {
  const run1 = await api.run(S.sid, "run1");
  const c1 = run1.config;
  const toggles = { ...(S.run2Toggles || {}) };
  if (!Object.keys(toggles).length) {
    for (const f of c1.selected_fields) toggles[f] = false;
    Object.assign(toggles, c1.blocking_toggles);
  }
  const waterfall = waterfallPanel();
  const results = h("div", {});
  const status = h("div");

  const refresh = debounce(async () => {
    S.run2Toggles = { ...toggles };
    try {
      waterfall.draw(await api.waterfall(S.sid, toggles));
    } catch (error) {
      fill(waterfall.node, note("bad", error.message));
    }
  }, 120);

  const run2 = h("button", { class: "primary" }, "Run 2");
  const drawResults = async () => {
    const [r2, cmp] = await Promise.all([renderRun("run2"), api.compare(S.sid)]);
    const hd = cmp.headline;
    fill(results, 
      h("hr"), h("h2", {}, "Run 2 results"), r2, h("hr"), h("h2", {}, "Run 1 vs Run 2"),
      metrics([["Edges", fmt(hd.edges[1]), hd.edges[1] - hd.edges[0]], ["Clusters", fmt(hd.clusters[1]), hd.clusters[1] - hd.clusters[0]],
        ["Mean match probability", num4(hd.mean_probability[1]), hd.mean_probability[0] == null ? null : hd.mean_probability[1] - hd.mean_probability[0]]]),
      cmp.accuracy.length ? h("div", {}, h("h3", {}, "Accuracy against ground truth"),
        table(cmp.accuracy.map((a) => ({ measure: a.measure, run1: a.run1, run2: a.run2, change: a.change })))) : null,
      h("h3", {}, "What changed between the runs"), table(cmp.changes), plots(cmp.charts),
      assistantBlock(), reportBlock(["run1", "run2"]));
  };
  run2.onclick = async () => {
    if (!Object.values(toggles).some(Boolean)) { fill(status, note("bad", "Switch on at least one blocking rule.")); return; }
    run2.disabled = true;
    const progress = stages(runStages(false, c1.linkage_type));
    fill(status, progress.node);
    try {
      await runAndWait(S.sid, "run2", {
        fields: c1.selected_fields, blocking_toggles: toggles, blocking_mode: c1.blocking_mode, operation_mode: c1.operation_mode,
        linkage_type: c1.linkage_type, hyperparams: c1.hyperparams, cluster_threshold: c1.cluster_threshold, comp_types: S.cfg?.comp_types,
      }, (s) => progress.update(s.stage, s.elapsed_s));
      S.runs.run2 = true;
      save();
      fill(status);
      await drawResults();
    } catch (error) {
      fill(status, note("bad", error.message));
    } finally { run2.disabled = false; }
  };

  put(view,
    h("h1", {}, "Compare runs"),
    h("p", {}, "Change the blocking rules and run again to see what the change does to the links. Everything else stays as in Run 1."),
    h("p", { class: "muted" }, "Candidate pairs are what blocking hands to the model to score; edges are the pairs it keeps as likely matches. A new rule can add many candidates and few or no edges."),
    metrics([["Run 1: edges", fmt(run1.cards.edges)], ["Run 1: clusters", fmt(run1.cards.clusters)]]),
    h("h2", {}, "Blocking rules for Run 2"), ruleEditor(run1, toggles, refresh),
    h("h2", {}, "Effect of your changes, live"), waterfall.node,
    h("div", { class: "row", style: "margin-top:18px" }, run2), status, results,
    h("div", { class: "row", style: "margin-top:22px" }, h("button", { class: "primary", onclick: () => go("export") }, "Continue to export")));
  refresh();
  if (S.runs.run2) await drawResults();
}

export async function renderExport(view) {
  const slots = ["run1", ...(S.runs.run2 ? ["run2"] : [])];
  let slot = "run1";
  const info = h("div", {});
  const draw = async () => {
    const [r, raw] = await Promise.all([api.run(S.sid, slot), api.raw(S.sid, slot)]);
    fill(info, 
      metrics([["Total records", fmt(r.cards.records)], ["Distinct cluster IDs", fmt(r.cards.clusters)],
        ["Records with a cluster", fmt(raw.n_cluster)]]),
      h("h3", {}, "Preview"), table(raw.cluster, null, { max: 50 }),
      h("div", { class: "row", style: "margin-top:16px" }, download(api.download(S.sid, slot, "cohort.csv"), `Download cohort CSV (${slot === "run1" ? "Run 1" : "Run 2"})`, "btn primary")));
  };
  put(view, h("h1", {}, "Export cohort"),
    h("p", {}, "Download the cohort as a CSV: every input column plus cluster_id. Records sharing a cluster_id are predicted to be the same real-world entity."),
    slots.length > 1 ? h("fieldset", {}, h("legend", {}, "Which run"), h("div", { class: "row" }, slots.map((s) => h("label", {}, h("input", { type: "radio", name: "exp", value: s, checked: s === slot,
      onchange: () => { slot = s; draw(); } }), s === "run1" ? "Run 1" : "Run 2")))) : null,
    info, h("p", { class: "muted" }, "The export is built on the server from the run's clusters; nothing leaves this session unless you download it."));
  await draw();
}
