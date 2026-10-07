// Shared state for the page-per-step app.

export const S = { sid: null, flow: "standard", overview: null, cfg: null, model: null, assistantUrl: "", runs: {} };
export const KEY = "cohort-builder-session";

export function save() {
  try { sessionStorage.setItem(KEY, JSON.stringify({ sid: S.sid, flow: S.flow, cfg: S.cfg })); } catch { /* storage is optional */ }
}

export function go(step) { location.hash = `#/${step}`; }

export function defaultConfig(o) {
  const d = o.defaults;
  return {
    fields: [...d.fields], blocking_toggles: { ...d.blocking_toggles }, blocking_mode: "OR",
    operation_mode: "dedupe", linkage_type: "probabilistic", cluster_threshold: 0.8,
    hyperparams: { max_iterations: 25, em_convergence: 0.0001, recall_estimate: 0.6 }, comp_types: { ...d.comp_types },
  };
}

// A new dataset: forget the old runs and start the configuration from its defaults.
export function setSession(overview, flow) {
  S.sid = overview.session_id;
  S.overview = overview;
  S.flow = flow ?? S.flow;
  S.runs = {};
  S.model = null;
  S.run2Toggles = null;
  S.cfg = defaultConfig(overview);
  save();
}
