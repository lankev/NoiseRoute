import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import QFTGate

from .circuit import normalize


def builtin(family="random", n=5, seed=0, depth=8):
    if n < 2:
        raise ValueError("At least two qubits")
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"{family}_{n}_s{seed}")
    if family == "qft":
        qc.append(QFTGate(n), range(n))
    elif family == "grover":
        qc.h(range(n))
        for _ in range(max(1, min(3, int(np.sqrt(n))))):
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            qc.h(range(n))
            qc.x(range(n))
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            qc.x(range(n))
            qc.h(range(n))
    elif family == "ansatz":
        for _ in range(depth):
            for q in range(n):
                qc.ry(float(rng.uniform(-np.pi, np.pi)), q)
                qc.rz(float(rng.uniform(-np.pi, np.pi)), q)
            for q in range(n - 1):
                qc.cx(q, q + 1)
            qc.cx(n - 1, 0)
    elif family == "arithmetic":
        qc.h(range(n))
        for q in reversed(range(1, n)):
            qc.mcx(list(range(q)), q)
        qc.x(0)
    elif family == "random":
        for _ in range(depth):
            for q in range(n):
                qc.ry(float(rng.uniform(-np.pi, np.pi)), q)
            order = rng.permutation(n)
            for i in range(0, n - 1, 2):
                qc.cx(int(order[i]), int(order[i + 1]))
    else:
        raise ValueError(family)
    return normalize(qc)


def save_circuit(qc, root, family, source, seed=0, split="test"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    text = qasm2.dumps(normalize(qc))
    # Hash and persist the same UTF-8 bytes: text-mode writes translate LF to
    # CRLF on Windows and would otherwise invalidate the manifest immediately.
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = root / f"{family}_{qc.num_qubits}_{digest[:12]}.qasm"
    path.write_bytes(payload)
    return dict(
        path=path.name,
        sha256=digest,
        family=family,
        source=source,
        qubits=qc.num_qubits,
        seed=seed,
        split=split,
    )


def prepare(
    root,
    sizes=(5, 10),
    seeds=(100000,),
    families=("qft", "grover", "ansatz", "random", "arithmetic"),
    mqt=False,
    qasmbench=None,
    max_qubits=20,
    split="test",
):
    records, errors = [], []
    for n in sizes:
        for family in families:
            source = "mqt.bench" if mqt else "builtin"
            for seed in seeds:
                try:
                    if mqt:
                        from mqt.bench import BenchmarkLevel, get_benchmark

                        upstream = {
                            "ansatz": "vqe_su2",
                            "grover": "grover",
                            "arithmetic": "cdkm_ripple_carry_adder",
                            "random": "randomcircuit",
                        }
                        mqt_n = max(4, n + n % 2) if family == "arithmetic" else n
                        qc = get_benchmark(
                            upstream.get(family, family), BenchmarkLevel.ALG, mqt_n, random_parameters=False
                        )
                        if qc.parameters:
                            rng = np.random.default_rng(seed)
                            qc = qc.assign_parameters(
                                {p: float(rng.uniform(-np.pi, np.pi)) for p in qc.parameters}
                            )
                    else:
                        qc = builtin(family, n, seed)
                    records.append(save_circuit(qc, root, family, source, seed, split))
                except Exception as exc:
                    errors.append(
                        dict(source=source, family=family, n=n, error=f"{type(exc).__name__}: {exc}")
                    )
    if qasmbench:
        for path in sorted(Path(qasmbench).rglob("*.qasm")):
            try:
                qc = qasm2.load(
                    path,
                    include_path=(str(path.parent), str(qasmbench)),
                    custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                )
                if qc.num_qubits <= max_qubits:
                    records.append(save_circuit(qc, root, path.stem, f"QASMBench:{path.name}", split=split))
            except Exception as exc:
                errors.append(dict(source=str(path), error=f"{type(exc).__name__}: {exc}"))
    records = list({r["sha256"]: r for r in records}.values())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(dict(circuits=records, errors=errors), indent=2))
    return records, errors


def load_manifest(path, split=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    out = []
    for r in manifest["circuits"]:
        if split is not None and r["split"] != split:
            continue
        source = path.parent / r["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Circuit hash mismatch: {source}")
        out.append((r, qasm2.load(source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)))
    if not out:
        raise ValueError(f"No circuits for split={split}")
    return out
