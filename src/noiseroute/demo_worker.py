"""Run native quantum libraries outside Streamlit's rerun threads."""

import json
import sys
from pathlib import Path

from qiskit import qasm2

from .benchmark import compile_one
from .datasets import builtin
from .device import synthetic


def main():
    root = Path(sys.argv[1])
    config = json.loads((root / "input.json").read_text())
    qc = (
        qasm2.loads(config["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        if config.get("qasm")
        else builtin(config["family"], config["n"], config["seed"], 3)
    )
    if not 2 <= qc.num_qubits <= 12:
        raise ValueError("La démo accepte 2 à 12 qubits logiques.")
    if not qc.count_ops().get("measure", 0):
        qc.measure_all()
    device = synthetic(config["topology"], qc.num_qubits, 7)
    checkpoint = config.get("checkpoint")
    algorithms = ["sabre", "noise_sabre", "learned" if checkpoint else "noiseroute"]
    rows = []
    for algorithm in algorithms:
        compiled, metrics = compile_one(qc, device, algorithm, config["seed"], checkpoint)
        rows.append(
            dict(algorithm=algorithm, metrics=metrics, drawing=str(compiled.draw(output="text", fold=70)))
        )
    (root / "output.json").write_text(
        json.dumps(dict(logical=qc.num_qubits, physical=device.n, edges=len(device.edges), rows=rows))
    )


if __name__ == "__main__":
    main()
