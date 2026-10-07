// The first screen's one moment: two small lists and the threads a linkage draws between them.
// Hover or focus a record to see why it matches; slide the threshold to see which pairs are linked.
// Records are invented for illustration.

import { h } from "./ui.js";

const A = [
  ["Maria Okafor", "12 Mar 1984 · Leeds"], ["James Whitfield", "3 Nov 1971 · York"], ["Priya Raman", "27 Jul 1990 · Bath"],
  ["Tomasz Nowak", "9 Jan 1965 · Hull"], ["Chloe Dubois", "18 May 1998 · Ely"], ["Samuel Adeyemi", "30 Sep 1979 · Derby"],
];
const B = [
  ["Sam Adeyemi", "30 Sep 1979 · Derby"], ["Tomasz Novak", "9 Jan 1965 · Hull"], ["Marria Okafor", "12 Mar 1984 · Leeds"],
  ["Lena Fischer", "2 Feb 2001 · Ely"], ["Priya Raman", "17 Jul 1990 · Bath"], ["Jim Whitfield", "3 Nov 1971 · York"],
];   // a different order from A, as real files are: the threads find the pairs anyway
// a, b = row index in each list; p = match probability; checks = how each field compares
const LINKS = [
  { a: 0, b: 2, p: 0.97, checks: [["Name", "close"], ["Date of birth", "same"], ["City", "same"]] },
  { a: 1, b: 5, p: 0.91, checks: [["Name", "close"], ["Date of birth", "same"], ["City", "same"]] },
  { a: 2, b: 4, p: 0.88, checks: [["Name", "same"], ["Date of birth", "close"], ["City", "same"]] },
  { a: 3, b: 1, p: 0.84, checks: [["Name", "close"], ["Date of birth", "same"], ["City", "same"]] },
  { a: 5, b: 0, p: 0.69, checks: [["Name", "close"], ["Date of birth", "same"], ["City", "same"]] },
  { a: 4, b: 3, p: 0.31, checks: [["Name", "differs"], ["Date of birth", "differs"], ["City", "same"]] },
];
const WORDS = { same: "same", close: "close", differs: "differs" };

export function heroDemo() {
  let threshold = 0.8;
  let hot = null;
  let first = true;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "threads");
  svg.setAttribute("aria-hidden", "true");
  const detail = h("div", { class: "demo-detail", "aria-live": "polite" });
  const summary = h("output", {});
  const demo = h("div", { class: "demo" });

  const rec = (side, rows) => rows.map(([name, line], i) => {
    const el = h("button", { class: `rec ${side}`, type: "button", "data-i": i }, h("b", {}, name), h("small", {}, line));
    const on = () => focus(side, i);
    el.addEventListener("pointerenter", on);
    el.addEventListener("focus", on);
    el.addEventListener("pointerleave", () => focus(null));
    el.addEventListener("blur", () => focus(null));
    return el;
  });
  const left = rec("a", A), right = rec("b", B);
  const cols = h("div", { class: "demo-cols" },
    h("div", {}, h("h4", {}, h("span", { class: "dot a" }), "Dataset A"), h("div", { class: "demo-list" }, left)),
    h("div", {}, h("h4", {}, h("span", { class: "dot b" }), "Dataset B"), h("div", { class: "demo-list" }, right)));
  cols.append(svg);

  const slider = h("input", { type: "range", min: 0.5, max: 0.99, step: 0.01, value: threshold, "aria-label": "Match threshold" });
  slider.addEventListener("input", () => { threshold = Number(slider.value); draw(); focus(null); });
  demo.append(cols, detail, h("div", { class: "demo-foot" }, h("label", {}, "Match threshold", slider), summary));

  function paths() {
    svg.replaceChildren();
    const box = cols.getBoundingClientRect();
    LINKS.forEach((link, i) => {
      const a = left[link.a].getBoundingClientRect(), b = right[link.b].getBoundingClientRect();
      const x1 = a.right - box.left, y1 = a.top + a.height / 2 - box.top, x2 = b.left - box.left, y2 = b.top + b.height / 2 - box.top;
      const dx = (x2 - x1) * 0.55;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M${x1} ${y1} C${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`);
      const linked = link.p >= threshold;
      path.setAttribute("class", `thread${linked ? "" : " weak"}${first && linked ? " draw" : ""}${hot === i ? " hot" : ""}`);
      if (first && linked) path.style.animationDelay = `${i * 110}ms`;
      path.dataset.i = i;
      svg.append(path);
    });
  }

  function focus(side, index) {
    hot = side == null ? null : LINKS.findIndex((l) => (side === "a" ? l.a === index : l.b === index));
    if (hot === -1) hot = null;
    demo.classList.toggle("focus", side != null);
    left.forEach((el, i) => el.classList.toggle("hot", side != null && (side === "a" ? i === index : hot != null && LINKS[hot].a === i)));
    right.forEach((el, i) => el.classList.toggle("hot", side != null && (side === "b" ? i === index : hot != null && LINKS[hot].b === i)));
    svg.querySelectorAll(".thread").forEach((p) => p.classList.toggle("hot", hot != null && Number(p.dataset.i) === hot));
    explain(side, index);
  }

  function explain(side, index) {
    if (side == null) {
      detail.replaceChildren(h("p", { class: "muted" }, "Hover or focus a record to see why it matches. Solid threads are links the model is confident in; dotted ones are candidates it leaves apart."));
      return;
    }
    const link = hot != null ? LINKS[hot] : null;
    if (!link) { detail.replaceChildren(h("p", {}, "No candidate in the other list: this record stays on its own.")); return; }
    const [an, al] = A[link.a], [bn, bl] = B[link.b];
    const [ad, ac] = al.split(" · "), [bd, bc] = bl.split(" · ");
    const values = [[an, bn], [ad, bd], [ac, bc]];
    detail.replaceChildren(
      h("p", {}, h("strong", {}, `Match probability ${Math.round(link.p * 100)}%`), link.p >= threshold ? ": linked as one person." : ": below the threshold, left apart."),
      h("table", {}, h("tbody", {}, link.checks.map(([field, verdict], i) => h("tr", {},
        h("th", { scope: "row" }, field), h("td", {}, values[i][0]), h("td", {}, values[i][1]),
        h("td", {}, h("span", { class: `pill ${verdict}` }, WORDS[verdict])))))));
  }

  function draw() {
    paths();
    const linked = LINKS.filter((l) => l.p >= threshold).length;
    summary.textContent = `${linked} linked, ${LINKS.length - linked} left apart`;
  }

  explain(null);
  requestAnimationFrame(() => { draw(); first = false; });
  if (window.ResizeObserver) new ResizeObserver(() => { if (!first) draw(); }).observe(cols);
  return demo;
}
