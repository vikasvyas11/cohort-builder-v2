// Step 1: choose where the data comes from. Standard (built-in data), Upload (your files), Advanced (a saved model).

import { api } from "./api.js";
import { heroDemo } from "./hero.js";
import { S, go, setSession } from "./state.js";
import { fmt, h, note, put, fill } from "./ui.js";

const VOTER_SIZES = [1200, 10000, 50000];

// Runs `work` with the button disabled and a status line; errors appear next to the button.
function busy(button, status, label, work) {
  return async () => {
    button.disabled = true;
    fill(status, h("span", { class: "muted", role: "status" }, label));
    try {
      await work();
    } catch (error) {
      fill(status, note("bad", error.message));
    } finally {
      button.disabled = false;
    }
  };
}

function standardCard() {
  const status = h("div");
  const size = h("select", { "aria-label": "Number of voters in Dataset A" },
    VOTER_SIZES.map((n) => h("option", { value: n }, `${fmt(n)} in A, ${fmt(n / 2)} in B`)));
  const people = h("button", { class: "primary" }, "Use people dataset");
  const voters = h("button", { class: "primary" }, "Use North Carolina voter dataset");
  people.onclick = busy(people, status, "Building the people dataset...", async () => {
    setSession(await api.createDemo("people", 0), "standard");
    go("profile");
  });
  voters.onclick = busy(voters, status, "Building the voter dataset...", async () => {
    setSession(await api.createDemo("voters", Number(size.value)), "standard");
    go("profile");
  });
  return h("section", { class: "card main-way", id: "start" },
    h("h2", {}, "Try it on built-in data"),
    h("p", { class: "muted" }, "Each dataset comes with a damaged copy (Dataset B) and a hidden ground truth, so you can see how accurate the linkage is. No real person is in either."),
    h("div", { class: "datasets" },
      h("div", { class: "dataset" }, h("h3", {}, "People"),
        h("p", {}, "1,000 UK-style records with duplicates: name, date of birth, city, email and postcode."), people),
      h("div", { class: "dataset" }, h("h3", {}, "North Carolina voters"),
        h("p", {}, "Synthetic voters whose demographics, places and common names follow the state's voter file. Pick a size to test scale, up to 50,000."),
        size, voters)),
    status);
}

function uploadCard() {
  const status = h("div");
  const fileA = h("input", { type: "file", accept: ".csv,.txt,.tsv", "aria-label": "Dataset A file" });
  const urlA = h("input", { type: "url", placeholder: "or a public https:// link", "aria-label": "Dataset A URL" });
  const idA = h("input", { type: "text", placeholder: "ID column (optional)", "aria-label": "Dataset A ID column" });
  const fileB = h("input", { type: "file", accept: ".csv,.txt,.tsv", "aria-label": "Dataset B file" });
  const urlB = h("input", { type: "url", placeholder: "or a public https:// link", "aria-label": "Dataset B URL" });
  const idB = h("input", { type: "text", placeholder: "ID column (optional)", "aria-label": "Dataset B ID column" });
  const bBox = h("div", { class: "stack", hidden: true }, h("div", { class: "row" }, fileB, urlB), idB);
  const modes = [["dedupe_only", "Find duplicates in Dataset A only"], ["uploaded", "Upload a Dataset B to link with"],
    ["sample", "Generate a damaged copy of A as Dataset B, to test linking"]];
  const radios = modes.map(([value, label], i) => h("label", {},
    h("input", { type: "radio", name: "bmode", value, checked: i === 0, onchange: () => { bBox.hidden = value !== "uploaded"; } }), label));
  const go1 = h("button", { class: "primary" }, "Upload and clean");
  go1.onclick = busy(go1, status, "Reading and cleaning the data...", async () => {
    const mode = radios.map((r) => r.querySelector("input")).find((i) => i.checked).value;
    const form = new FormData();
    if (fileA.files[0]) form.append("file_a", fileA.files[0]);
    form.append("url_a", urlA.value);
    form.append("id_col_a", idA.value.trim());
    form.append("mode", mode);
    if (mode === "uploaded") {
      if (fileB.files[0]) form.append("file_b", fileB.files[0]);
      form.append("url_b", urlB.value);
      form.append("id_col_b", idB.value.trim());
    }
    setSession(await api.upload(form), "upload");
    go("profile");
  });
  return h("section", { class: "card stack" },
    h("h2", {}, "Use your own files"),
    h("p", { class: "muted" }, "CSV or tab-delimited text, up to 100 MB each. Field names, empty rows, duplicates and dates are cleaned for you. Links must be public http(s)."),
    h("div", { class: "stack" }, h("strong", {}, "Dataset A"), h("div", { class: "row" }, fileA, urlA), idA),
    h("div", { class: "stack" }, h("strong", {}, "What to do"), ...radios, bBox), go1, status);
}

function advancedCard() {
  const status = h("div");
  const file = h("input", { type: "file", accept: ".json", "aria-label": "Splink model JSON" });
  const source = h("select", { "aria-label": "Dataset to predict on" },
    h("option", { value: "people" }, "People dataset"),
    VOTER_SIZES.map((n) => h("option", { value: n }, `Voter dataset, ${fmt(n)} in A`)));
  const go1 = h("button", {}, "Load model and dataset");
  go1.onclick = busy(go1, status, "Loading...", async () => {
    if (!file.files[0]) throw new Error("Choose a model JSON file first.");
    const people = source.value === "people";
    const overview = await api.createDemo(people ? "people" : "voters", people ? 0 : Number(source.value));
    const summary = await api.model(overview.session_id, file.files[0]);
    setSession(overview, "advanced");
    S.model = summary;
    S.cfg.linkage_type = summary.detected_linkage_type;
    go("profile");
  });
  return h("section", { class: "card stack" },
    h("h2", {}, "Skip training with a saved model"),
    h("p", { class: "muted" }, "Load a Splink 4 model JSON, from the Results step of this app or from linker.misc.save_model_to_json(), and it predicts straight away."),
    file, source, go1, status);
}

export async function renderData(view) {
  put(view,
    h("div", { class: "hero" },
      h("div", {},
        h("h1", {}, "Find the same person across messy records"),
        h("p", {}, "Link two files, or find duplicates inside one. Then check how accurate the match was, compare a second attempt and export the cohort. Built on Splink and DuckDB, tested to 50,000 records."),
        h("a", { class: "btn primary", href: "#start", onclick: (e) => { e.preventDefault(); document.getElementById("start")?.scrollIntoView({ behavior: "smooth" }); } }, "Start with built-in data")),
      heroDemo()),
    h("div", { class: "ways" }, standardCard(), uploadCard(), advancedCard()));
}
