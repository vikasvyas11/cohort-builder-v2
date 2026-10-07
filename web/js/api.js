// Calls to the Cohort Builder API. Errors carry the server's plain-language message.

export const BASE = (window.COHORT_API || "").replace(/\/$/, "");
export const url = (path) => `${BASE}${path}`;

async function call(path, options = {}) {
  let response;
  try {
    response = await fetch(url(path), options);
  } catch {
    throw new Error("Could not reach the server. If it has been idle it may be waking up: wait a moment and try again.");
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try { detail = (await response.json()).detail || detail; } catch { /* keep the generic message */ }
    throw new Error(detail);
  }
  return response;
}

const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  config: () => call("/api/config").then((r) => r.json()),
  createDemo: (source, nRows) => call("/api/sessions", json({ source, n_rows: nRows })).then((r) => r.json()),
  upload: (form) => call("/api/sessions/upload", { method: "POST", body: form }).then((r) => r.json()),
  overview: (sid) => call(`/api/sessions/${sid}`).then((r) => r.json()),
  datasetB: (sid, body) => call(`/api/sessions/${sid}/dataset-b`, json(body)).then((r) => r.json()),
  cohort: (sid, filters) => call(`/api/sessions/${sid}/cohort`, json(filters)).then((r) => r.json()),
  model: (sid, file) => {
    const form = new FormData();
    form.append("file", file);
    return call(`/api/sessions/${sid}/model`, { method: "POST", body: form }).then((r) => r.json());
  },
  estimate: (sid, cfg) => call(`/api/sessions/${sid}/estimate`, json(cfg)).then((r) => r.json()),
  startRun: (sid, slot, cfg) => call(`/api/sessions/${sid}/runs/${slot}`, json(cfg)).then((r) => r.json()),
  job: (id) => call(`/api/jobs/${id}`).then((r) => r.json()),
  run: (sid, slot) => call(`/api/sessions/${sid}/runs/${slot}`).then((r) => r.json()),
  demographics: (sid, slot) => call(`/api/sessions/${sid}/runs/${slot}/demographics`).then((r) => r.json()),
  explorer: (sid, slot, body) => call(`/api/sessions/${sid}/runs/${slot}/explorer`, json(body)).then((r) => r.json()),
  recluster: (sid, slot, body) => call(`/api/sessions/${sid}/runs/${slot}/recluster`, json(body)).then((r) => r.json()),
  raw: (sid, slot) => call(`/api/sessions/${sid}/runs/${slot}/raw`).then((r) => r.json()),
  studio: (sid, slot) => call(`/api/sessions/${sid}/runs/${slot}/studio`).then((r) => r.text()),
  waterfall: (sid, toggles) => call(`/api/sessions/${sid}/waterfall`, json({ toggles })).then((r) => r.json()),
  compare: (sid) => call(`/api/sessions/${sid}/compare`).then((r) => r.json()),
  download: (sid, slot, file) => url(`/api/sessions/${sid}/runs/${slot}/${file}`),
};

// Start a run, then poll until it finishes. `onStage` receives {stage, elapsed_s}.
export async function runAndWait(sid, slot, cfg, onStage) {
  const { job_id: id } = await api.startRun(sid, slot, cfg);
  for (;;) {
    const status = await api.job(id);
    onStage?.(status);
    if (status.status === "done") return;
    if (status.status === "error") throw new Error(status.error || "The run failed.");
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}
