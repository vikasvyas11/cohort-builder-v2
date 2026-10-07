// Router. One page per step; the three ways in (Standard, Upload, Advanced) all lead through
// the same Profile, Configure, Results, Compare and Export steps.

import { api } from "./api.js";
import { KEY, S, defaultConfig, go } from "./state.js";
import { clear, fmt, h, note, put } from "./ui.js";
import { renderData } from "./data.js";
import { renderProfile } from "./profile.js";
import { renderConfigure } from "./configure.js";
import { renderResults } from "./results.js";
import { renderCompare, renderExport } from "./compare.js";

const STEPS = [
  ["data", "Data", renderData], ["profile", "Profile", renderProfile], ["configure", "Configure", renderConfigure],
  ["results", "Results", renderResults], ["compare", "Compare", renderCompare], ["export", "Export", renderExport],
];

function reachable(step) {
  const has = !!S.sid && !!S.overview;
  return { data: true, profile: has, configure: has, results: has && !!S.runs.run1, compare: has && !!S.runs.run1,
    export: has && !!S.runs.run1 }[step];
}

// The rail is the progress thread: finished steps fill in, the current one is ringed.
function drawRail(current) {
  const at = STEPS.findIndex(([id]) => id === current);
  const list = clear(document.getElementById("stepper"));
  STEPS.forEach(([id, label], i) => {
    const done = i < at && reachable(id);
    list.append(h("li", {}, h("button", {
      class: done ? "done" : "", disabled: !reachable(id), "aria-current": id === current ? "step" : null, onclick: () => go(id),
    }, h("span", { class: "node" }, done ? "" : String(i + 1)), h("span", { class: "label" }, label))));
  });
  const o = S.overview;
  const ctx = clear(document.getElementById("context"));
  if (!o) return;
  put(ctx,
    h("strong", {}, { people: "People dataset", voters: "Voter dataset", upload: "Your upload" }[o.source] || "Dataset"),
    h("div", { class: "pair" }, h("span", { class: "dot a" }), `Dataset A: ${fmt(o.rows_a)} records`),
    h("div", { class: "pair" }, h("span", { class: "dot b" }), o.rows_b == null ? "Dataset B: not set" : `Dataset B: ${fmt(o.rows_b)} records`),
    S.runs.run1 ? h("div", { class: "pair" }, h("span", { class: "dot link" }), S.runs.run2 ? "Run 1 and Run 2 done" : "Run 1 done") : null);
}

async function route() {
  const view = document.getElementById("view");
  const step = (location.hash.replace(/^#\/?/, "") || "data").split("/")[0];
  const entry = STEPS.find(([id]) => id === step) || STEPS[0];
  if (!reachable(entry[0])) { go(S.sid ? "configure" : "data"); return; }
  drawRail(entry[0]);
  clear(view);
  try {
    await entry[2](view);
  } catch (error) {
    view.append(note("bad", error.message));
  }
  window.scrollTo({ top: 0 });
}

async function start() {
  try { S.assistantUrl = (await api.config()).assistant_url; } catch { /* the link is optional */ }
  try {
    const saved = JSON.parse(sessionStorage.getItem(KEY) || "null");
    if (saved?.sid) {
      S.overview = await api.overview(saved.sid);
      S.sid = saved.sid;
      S.flow = saved.flow || "standard";
      S.cfg = saved.cfg || defaultConfig(S.overview);
      for (const slot of S.overview.runs || []) S.runs[slot] = true;
    }
  } catch { sessionStorage.removeItem(KEY); }          // the session expired: start again
  window.addEventListener("hashchange", route);
  route();
}

start();
