"""IBM local routing adapter for a separate, pinned environment."""

import importlib.metadata
import json
import sys
import time
from pathlib import Path

from qiskit import qasm2, qpy
from qiskit_ibm_transpiler import generate_ai_pass_manager

from .device import Device
from .metrics import evaluate


def main():
    root, seed = Path(sys.argv[1]), int(sys.argv[2])
    qc = qasm2.load(root / "input.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    device = Device.load(root / "device.json")
    swaps = 0

    def callback(**kw):
        nonlocal swaps
        if kw["pass_"].__class__.__name__ == "AIRouting":
            swaps = kw["dag"].count_ops().get("swap", 0)

    # This upstream helper explicitly constructs AIRouting(local_mode=True).
    pm = generate_ai_pass_manager(
        coupling_map=device.coupling,
        ai_optimization_level=3,
        optimization_level=3,
        ai_layout_mode="optimize",
        include_ai_synthesis=False,
        qiskit_transpile_options={"target": device.target(), "seed_transpiler": seed},
    )
    start = time.perf_counter()
    output = pm.run(qc, callback=callback)
    metrics = evaluate(output, device)
    metrics.update(
        compile_ms=(time.perf_counter() - start) * 1000,
        swaps_added=swaps,
        fallback_steps=0,
        seed_controls_router=False,
        ai_qiskit=importlib.metadata.version("qiskit"),
        ai_version=importlib.metadata.version("qiskit-ibm-transpiler"),
    )
    with (root / "output.qpy").open("wb") as stream:
        qpy.dump(output, stream)
    (root / "metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
