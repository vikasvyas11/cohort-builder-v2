// Step 2: what is in the data before any linking happens.

import { api } from "./api.js";
import { S, defaultConfig, go, save } from "./state.js";
import { fmt, h, metrics, note, pct, plots, table, put, fill } from "./ui.js";

function edaSection(eda) {
  const log = eda?.log;
  if (!log || !log.summary) return null;
  const s = log.summary;
  const nulls = log.null_rows_removed || {};
  const steps = [
    ["Field names standardised", Object.keys(log.field_names?.changed || {}).length],
    ["Columns dropped (no values)", (log.null_columns_dropped || []).length],
    ["Rows removed (no values)", nulls.all_empty_rows || 0],
    ["Rows removed (one value only)", nulls.one_value_rows || 0],
    ["Rows removed (two values only)", nulls.two_value_rows || 0],
    ["Duplicate rows removed", log.duplicates_removed || 0],
    ["Date columns standardised", Object.keys(log.dates_standardised || {}).length],
  ].map(([step, n]) => ({ step, n }));
  const corr = eda.high_correlation || [];
  return h("details", { class: "card" },
    h("summary", {}, `Cleaning log: ${fmt(s.original_rows)} rows in, ${fmt(s.final_rows)} kept`),
    table(steps),
    corr.length ? h("div", {},
      h("h3", {}, "Near-duplicate columns"),
      h("p", { class: "muted" }, "These pairs agree row for row, so one of each adds nothing. Consider leaving one out on the Configure step."),
      table(corr.map((c) => ({ column_a: c.a, column_b: c.b, agreement: c.agreement })))) : null);
}

function cohortFilter(o) {
  const schema = o.cohort_filter;
  if (!schema) return null;
  const status = h("div");
  const boxes = {};
  const sets = schema.categorical.map((c) => h("fieldset", {}, h("legend", {}, c.field),
    h("div", { class: "chips" }, c.options.map((opt) => {
      const box = h("input", { type: "checkbox", value: opt.value });
      (boxes[c.field] ||= []).push(box);
      return h("label", { class: "chip" }, box, h("span", {}, opt.label === opt.value ? opt.value : `${opt.label} (${opt.value})`));
    }))));
  const ranges = schema.ranges.map((r) => {
    const lo = h("input", { type: "number", value: r.bounds[0], "aria-label": `${r.field} minimum` });
    const hi = h("input", { type: "number", value: r.bounds[1], "aria-label": `${r.field} maximum` });
    return { field: r.field, bounds: r.bounds, lo, hi, node: h("div", { class: "row" }, h("strong", {}, r.field), lo, "to", hi) };
  });
  const apply = h("button", {}, "Apply cohort filter");
  const reset = h("button", { class: "link" }, "Clear filter");
  const send = async (filters) => {
    apply.disabled = reset.disabled = true;
    try {
      const result = await api.cohort(S.sid, filters);
      S.overview = result;
      S.runs = {};
      S.cfg = defaultConfig(result);
      save();
      window.dispatchEvent(new HashChangeEvent("hashchange"));   // redraw this step with the filtered data
    } catch (error) {
      fill(status, note("bad", error.message));
      apply.disabled = reset.disabled = false;
    }
  };
  apply.onclick = () => send({
    categorical: Object.fromEntries(Object.entries(boxes).map(([f, bs]) => [f, bs.filter((b) => b.checked).map((b) => b.value)])),
    ranges: Object.fromEntries(ranges.map((r) => [r.field, [Number(r.lo.value), Number(r.hi.value)]])),
    birth_state_include_missing: true,
  });
  reset.onclick = () => send({ categorical: {}, ranges: {} });
  return h("details", { class: "card" },
    h("summary", {}, `Cohort filter (${fmt(schema.rows_unfiltered)} records before filtering)`),
    h("p", { class: "muted" }, "Restrict Dataset A to a demographic cohort before linking. Filters combine with AND; leave a group empty for no restriction."),
    ...sets, ...ranges.map((r) => r.node), h("div", { class: "row" }, apply, reset), status);
}

export async function renderProfile(view) {
  const o = S.overview;
  put(view, 
    h("h1", {}, "Profile of your data"),
    h("p", { class: "muted" }, S.flow === "advanced" ? "Advanced flow: a saved model will predict on this data." : "A look at the data before linking."),
    metrics([["Records, Dataset A", fmt(o.rows_a)], ["Columns", fmt(o.columns)],
      ["Dataset B", o.rows_b == null ? "Not set" : `${fmt(o.rows_b)} records`],
      ["Ground truth", o.has_ground_truth ? "Available" : "None"]]),
    S.model ? note("ok", `Model loaded: ${S.model.comparisons} comparisons, ${S.model.blocking_rules} blocking rules (${S.model.detected_linkage_type}${S.model.from_marker ? "" : ", inferred"}).`) : null,
    edaSection(o.eda),
    cohortFilter(o),
    h("h2", {}, "Columns"),
    table(o.column_info.map((c) => ({ column: c.column, type: c.type, missing: pct(c.missing_pct), distinct_values: c.distinct })),
      ["column", "type", "missing", "distinct_values"]),
    h("h2", {}, "Dataset A preview"), table(o.preview_a.map((r) => r), null, { max: 8 }),
    o.demographics.length ? h("div", {}, h("h2", {}, "Demographic profile"), plots(o.demographics.map((d) => d.figure), "two"))
      : h("p", { class: "muted" }, "No recognised demographic columns in this dataset."),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => go("configure") }, "Continue to configure")));
}
