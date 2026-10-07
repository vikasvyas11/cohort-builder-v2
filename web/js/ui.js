// Small DOM helpers. Text is always set with textContent, never innerHTML, because
// column names and cell values come from the user's own data.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  add(el, children);
  return el;
}

function add(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

// Append children, skipping null and false (el.append would print the word "null").
export const put = (el, ...children) => { add(el, children); return el; };
export const clear = (el) => { el.replaceChildren(); return el; };
// Replace children, skipping null and false, so a conditional piece never prints the word "null".
export const fill = (el, ...children) => { el.replaceChildren(); add(el, children); return el; };
export const fmt = (n) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString("en-GB"));
export const pct = (x, d = 1) => (x == null ? "–" : `${Number(x).toFixed(d)}%`);
export const num4 = (x) => (x == null ? "–" : Number(x).toFixed(4));

// ── numbers and notes ─────────────────────────────────────────────────────────

export function metric(label, value, delta) {
  const d = delta == null || delta === 0 ? null
    : h("div", { class: `d ${delta > 0 ? "up" : "down"}` }, `${delta > 0 ? "+" : ""}${fmt(delta)}`);
  return h("div", { class: "cell" }, h("div", { class: "v" }, value), h("div", { class: "l" }, label), d);
}

export const metrics = (items) => h("div", { class: "strip" }, items.map((m) => metric(...m)));

export function note(kind, ...children) {
  return h("div", { class: `note ${kind}`, role: kind === "bad" ? "alert" : "status" }, ...children);
}

// A table from a list of row objects. `cols` = [key] or [[key, label]]; numbers are right aligned.
export function table(rows, cols, { max = 200 } = {}) {
  if (!rows || !rows.length) return h("p", { class: "muted" }, "Nothing to show.");
  const spec = (cols || Object.keys(rows[0])).map((c) => (Array.isArray(c) ? c : [c, c.replaceAll("_", " ")]));
  const numeric = Object.fromEntries(spec.map(([k]) => [k, typeof rows[0][k] === "number"]));
  const head = h("tr", {}, spec.map(([k, label]) => h("th", { scope: "col", class: numeric[k] ? "num" : "" }, label)));
  const body = rows.slice(0, max).map((r) =>
    h("tr", {}, spec.map(([k]) => {
      const v = r[k];
      const isNum = typeof v === "number";
      return h("td", { class: isNum ? "num" : "" }, v == null ? "" : isNum ? (Number.isInteger(v) ? fmt(v) : Number(v.toFixed(4)).toString()) : v);
    })));
  return h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

// ── controls ──────────────────────────────────────────────────────────────────

// A field or rule you can switch on: a pill with a filled dot when on.
export function chip(label, checked, onchange, { disabled = false } = {}) {
  return h("label", { class: "chip" },
    h("input", { type: "checkbox", checked: !!checked, disabled, onchange: (e) => onchange(e.target.checked) }),
    h("span", {}, label));
}

export function switchEl(checked, onchange, label) {
  return h("span", { class: "switch" },
    h("input", { type: "checkbox", role: "switch", checked: !!checked, "aria-label": label, onchange: (e) => onchange(e.target.checked) }), h("i"));
}

// A group of radio options laid out as selectable rows.
export function opts(name, options, current, onchange) {
  return h("div", { class: "opts" }, options.map(([value, label, disabled]) => h("label", {},
    h("input", { type: "radio", name, value, checked: value === current, disabled: !!disabled, onchange: () => onchange(value) }),
    h("span", {}, label))));
}

export const debounce = (fn, ms = 250) => {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

export function download(url, label, cls = "btn") {
  return h("a", { class: cls, href: url, download: "" }, label);
}

// ── charts ────────────────────────────────────────────────────────────────────

const dark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

// The server sends neutral figures; the page gives them the app's type and quiet grid.
function themed(fig, height) {
  const d = dark();
  const ink = d ? "#c4cfd8" : "#3a4550";
  const grid = d ? "rgba(255,255,255,.09)" : "rgba(23,33,43,.09)";
  const layout = { ...fig.layout, autosize: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Source Sans 3, system-ui, sans-serif", size: 12, color: ink }, hoverlabel: { font: { family: "Source Sans 3, system-ui, sans-serif" } } };
  delete layout.template;
  for (const axis of ["xaxis", "yaxis"]) {
    layout[axis] = { ...(layout[axis] || {}), gridcolor: grid, linecolor: grid, zerolinecolor: grid, automargin: true };
  }
  if (layout.title && typeof layout.title === "object") layout.title = { ...layout.title, font: { family: "Bricolage Grotesque, system-ui, sans-serif", size: 15, color: ink } };
  if (height) layout.height = height;
  return layout;
}

const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

// Draw a Plotly figure (JSON from the API) into a new div.
export function plot(fig, height) {
  const div = h("div", { class: "plot" });
  if (fig) queueMicrotask(() => window.Plotly.newPlot(div, fig.data, themed(fig, height), CONFIG));
  return div;
}

// A chart that is redrawn in place (no flicker) as new figures arrive: the live waterfall.
export function livePlot() {
  const div = h("div", { class: "plot" });
  let drawn = false;
  div.set = (fig, height) => {
    const layout = themed(fig, height);
    if (drawn) window.Plotly.react(div, fig.data, layout, CONFIG);
    else { drawn = true; window.Plotly.newPlot(div, fig.data, layout, CONFIG); }
  };
  return div;
}

export const plots = (figs, cls = "two") => h("div", { class: `grid ${cls}` }, (figs || []).filter(Boolean).map((f) => plot(f)));

export function tabs(spec, start = 0) {
  // spec = [[label, () => Node]]; panels render lazily, once.
  const bar = h("div", { class: "tabs", role: "tablist" });
  const body = h("div", { class: "tab-body" });
  const done = {};
  const buttons = spec.map(([label], i) => h("button", {
    role: "tab", "aria-selected": "false", onclick: () => show(i),
  }, label));
  bar.append(...buttons);
  function show(i) {
    buttons.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    if (!done[i]) done[i] = spec[i][1]();
    body.replaceChildren(done[i]);
    body.querySelectorAll(".plot").forEach((p) => window.Plotly && p.data && window.Plotly.Plots.resize(p));
  }
  show(start);
  return h("div", {}, bar, body);
}

// ── a run in progress ─────────────────────────────────────────────────────────

// The steps the server reports while it works, shown as a checklist that fills in.
export function stages(names) {
  const items = names.map((n) => h("li", {}, n));
  const line = h("p", { class: "muted", role: "status" }, "Starting...");
  const node = h("div", { "aria-live": "polite" }, h("ul", { class: "stages" }, items), line);
  const update = (stage, elapsed) => {
    const at = names.findIndex((n) => stage.startsWith(n));
    items.forEach((li, i) => { li.className = at < 0 ? "" : i < at ? "done" : i === at ? "now" : ""; });
    line.textContent = `${stage} (${elapsed}s)`;
  };
  return { node, update };
}

export const runStages = (fromModel, linkageType) => [
  "Counting candidate pairs",
  fromModel ? "Predicting with the uploaded model"
    : linkageType === "probabilistic" ? "Training the model and scoring pairs" : "Applying the exact-match rules",
  "Computing metrics", "Building the precision-recall curve", "Preparing the blocking explorer",
].filter((s) => !(fromModel && s === "Counting candidate pairs"));
