"""Isolated DAG/mapping microbenchmark; excludes features, policy and compilation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from noiseroute_core import RoutingCore

from noiseroute.core import PythonCore

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/core-benchmark.json")
args = p.parse_args()
rng = np.random.default_rng(41)
n = 50
edges = [(i, i + 1) for i in range(n - 1)]
gates = [rng.choice(n, 2, replace=False).tolist() for _ in range(500)]
actions = rng.integers(0, len(edges), 500).tolist()
results = {}
for cls in (PythonCore, RoutingCore):
    trials = []
    for _ in range(10):
        core = cls(n, edges, gates, list(range(n)))
        start = time.perf_counter_ns()
        for action in actions:
            core.swap(action)
            core.drain()
            core.mask()
            core.mapping()
        trials.append((time.perf_counter_ns() - start) / 1e6)
    results[cls.__name__] = dict(median_ms=float(np.median(trials)), steps=500)
results["scope"] = "Synthetic DAG microbenchmark only, not complete environment throughput."
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
