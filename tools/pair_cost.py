"""How much time and memory does a run cost as candidate pairs grow?

    python tools/pair_cost.py <rule,rule,...> [memory_cap_gb]

Runs one deduplication of 50,000 synthetic voters with exactly those blocking rules (write a
combined rule as a+b), prints the pair count, time and peak memory, and stops itself above the cap.
"""

import os
import sys
import time

os.environ["COHORT_BUILDER_MAX_CANDIDATE_PAIRS"] = "100000000"      # this tool measures; the app's own cap is separate
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.memory import guard, memory_mb  # noqa: E402

CAP_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0


guard(CAP_GB)

from modules import metrics_engine as me  # noqa: E402
from modules import splink_runner as sr  # noqa: E402
from modules.data_builder import load_voter_datasets  # noqa: E402
from modules.voter_data import LINKAGE_FIELDS  # noqa: E402

a, b, _, _ = load_voter_datasets(50_000)
fields = [f for f in LINKAGE_FIELDS if f in a.columns]
rules = {r: True for r in sys.argv[1].split(",")}
t = time.time()
pairs = sr.estimate_candidate_pairs(a, None, fields, rules, "OR", "dedupe")
print(f"candidate pairs (sum over rules): {pairs:,}   counted in {time.time() - t:.1f}s", flush=True)
t = time.time()
res = sr.run_linkage(a, None, fields, rules, "dedupe", "probabilistic")
print(f"run_linkage: {time.time() - t:.1f}s  edges kept {res['n_edges']:,}  mem {memory_mb()[0]} MB peak {memory_mb()[1]} MB", flush=True)
t = time.time()
cm = me.compute_confusion_matrix(res["df_predict"], a, None, "dedupe", 0.8)
curve = me.compute_threshold_curve(res["df_predict"], a, None, "dedupe")
cov = sr.build_coverage_matrix(res["df_predict"], fields)
print(f"metrics + curve + coverage: {time.time() - t:.1f}s  P={cm['precision']:.3f} R={cm['recall']:.3f}  "
      f"peak {memory_mb()[1]} MB", flush=True)
