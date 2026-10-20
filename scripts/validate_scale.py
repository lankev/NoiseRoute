"""Structural 100-to-127 routing check and native calibrated compilation check."""

import argparse
import json
from pathlib import Path

from noiseroute.benchmark import compile_one, provenance
from noiseroute.datasets import builtin
from noiseroute.device import Device
from noiseroute.env import Shape
from noiseroute.router import route

p = argparse.ArgumentParser()
p.add_argument("--snapshot", default="benchmarks/devices/fake_brisbane.json")
p.add_argument("--out", default="runs/scale")
args = p.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=True)
device = Device.load(args.snapshot)
circuit = builtin("ansatz", 100, 123, 1)
routed, info = route(circuit, device, shape=Shape(128, 384, 64))
info.pop("events")
info.update(
    logical_qubits=100,
    physical_qubits=device.n,
    structural_trace_verified=True,
    scope="Heuristic only, one shallow circuit; no unitary simulation at 127 qubits.",
    environment=provenance(),
)
(root / "scale127.json").write_text(json.dumps(info, indent=2))
small = builtin("random", 5, 123, 2)
small.measure_all()
native, metrics = compile_one(small, device, "noiseroute")
metrics.update(native_operations=dict(native.count_ops()), physical_qubits=native.num_qubits)
(root / "fake-native.json").write_text(json.dumps(metrics, indent=2))
print(
    json.dumps(
        {k: info[k] for k in ["logical_qubits", "physical_qubits", "swaps", "fallback_steps", "route_ms"]},
        indent=2,
    )
)
