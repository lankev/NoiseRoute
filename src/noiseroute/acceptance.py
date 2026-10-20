"""Fail-closed checkpoint selection on a fixed, small-circuit development corpus.

The gate compares per-case medians over >=20 compiler seeds. It is deliberately
not a claim about every seed, every circuit, physical fidelity or unseen devices.
"""

import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from qiskit import qpy
from qiskit.quantum_info import Operator

from .benchmark import compile_one, provenance, settings
from .datasets import builtin
from .device import synthetic
from .objectives import distinct_seeds

METRICS = {"log_esp": 1, "swaps_added": -1, "depth_2q": -1, "compile_ms": -1}


def validation_config(config):
    config = dict(config)
    config["seeds"] = distinct_seeds(config.get("seeds", list(range(20))))
    for field in ("sizes", "families", "topologies", "circuit_seeds"):
        values = config.get(field)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"validation.{field} must be a non-empty list without duplicates")
    if any(type(n) is not int or not 2 <= n <= 8 for n in config["sizes"]):
        raise ValueError("Acceptance currently requires 2..8 qubits for exact equivalence checks")
    if any(t not in ("line", "ring", "grid", "neutral") for t in config["topologies"]):
        raise ValueError("Acceptance supports equal-width line/ring/grid/neutral devices")
    if any(f not in ("random", "ansatz", "qft", "grover", "arithmetic") for f in config["families"]):
        raise ValueError("Unknown validation circuit family")
    distinct_seeds(config["circuit_seeds"], minimum=1)
    distinct_seeds([config.setdefault("device_seed", 177)], minimum=1)
    baselines = config.setdefault("baselines", ["sabre", "noise_sabre"])
    if (
        "sabre" not in baselines
        or len(set(baselines)) != len(baselines)
        or any(b not in ("sabre", "noise_sabre") for b in baselines)
    ):
        raise ValueError("validation.baselines must include sabre; optional noise_sabre")
    for key, default in (("every_steps", 4096), ("torch_threads", 1), ("circuit_depth", 4)):
        config.setdefault(key, default)
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"validation.{key} must be a positive integer")
    return config


def check_equivalence(source, compiled, device, atol=1e-7):
    """Exact terminal-measurement equivalence for ALL input states, n<=8.

    Level 3 can remove diagonal gates before measurement, so equality of the
    measured projectors (unitary rows up to independent phases) is appropriate.
    Input and output permutations are both accounted for; no sampled states.
    """
    n = source.num_qubits
    if n > 8 or compiled.num_qubits != n or device.n != n:
        raise ValueError("Exact validation needs matching circuit/device widths <=8")

    def measurement_map(qc):
        measured = {}
        for inst in qc.data:
            if inst.operation.name == "measure":
                c, q = qc.find_bit(inst.clbits[0]).index, qc.find_bit(inst.qubits[0]).index
                if c in measured:
                    raise ValueError("Repeated classical output")
                measured[c] = q
        if set(measured) != set(range(n)) or set(measured.values()) != set(range(n)):
            raise ValueError("Validation requires a full one-to-one terminal measurement")
        return [measured[i] for i in range(n)]

    if measurement_map(source) != list(range(n)):
        raise ValueError("Source validation circuits must measure q[i] into c[i]")
    output_mapping = measurement_map(compiled)
    initial = (
        compiled.layout.initial_index_layout(filter_ancillas=True) if compiled.layout else list(range(n))
    )
    if sorted(initial) != list(range(n)):
        raise ValueError("Invalid initial permutation")
    for inst in compiled.data:
        if len(inst.qubits) == 2:
            edge = tuple(sorted(compiled.find_bit(q).index for q in inst.qubits))
            if edge not in device.edges:
                raise ValueError("Compiled circuit violates device coupling")

    def permutation(mapping):
        return [sum(((i >> v) & 1) << p for v, p in enumerate(mapping)) for i in range(1 << n)]

    expected = Operator(source.remove_final_measurements(inplace=False)).data
    physical = Operator(compiled.remove_final_measurements(inplace=False)).data
    actual = physical[np.ix_(permutation(output_mapping), permutation(initial))]
    overlap = np.sum(expected.conj() * actual, axis=1)
    if not np.allclose(np.abs(overlap), 1, rtol=0, atol=atol) or not np.allclose(
        actual, overlap[:, None] * expected, rtol=0, atol=atol
    ):
        raise ValueError("Circuit measurement equivalence failed")


def assess(rows, cases, seeds, baselines):
    """No missing/invalid row can pass. Improvements are positive in all metrics."""
    seeds = distinct_seeds(seeds)
    reasons, comparisons = [], []
    if not cases or len(set(cases)) != len(cases) or not baselines or "sabre" not in baselines:
        return dict(accepted=False, reasons=["empty/invalid validation protocol"], comparisons=[])
    expected = {(c, a, s) for c in cases for a in ["learned", *baselines] for s in seeds}
    actual = [(r["case"], r["algorithm"], r["seed"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        reasons.append("missing, extra or duplicate benchmark rows")
    invalid = [
        r
        for r in rows
        if r.get("status") != "ok"
        or any(not isinstance(r.get(m), (int, float)) or not math.isfinite(r[m]) for m in METRICS)
        or r.get("equivalent") is not True
    ]
    if invalid:
        reasons.append(f"{len(invalid)} failed, non-finite or unverified rows")
    if reasons:
        return dict(accepted=False, reasons=reasons, comparisons=[])
    fallbacks = sum(r.get("fallback_steps") != 0 for r in rows)
    if fallbacks:
        reasons.append(f"{fallbacks} routing fallback rows; pure learned routing required")
    for case in cases:
        group = {
            a: [r for r in rows if r["case"] == case and r["algorithm"] == a] for a in ["learned", *baselines]
        }
        for baseline in baselines:
            metrics = {}
            for metric, direction in METRICS.items():
                candidate = float(np.median([r[metric] for r in group["learned"]]))
                reference = float(np.median([r[metric] for r in group[baseline]]))
                gain = direction * (candidate - reference)
                tolerance = 1e-9 if metric == "log_esp" else 0.0
                if gain < -tolerance:
                    reasons.append(f"{case}/{baseline}: regression in {metric}")
                metrics[metric] = dict(candidate=candidate, baseline=reference, gain=gain)
            comparisons.append(dict(case=case, baseline=baseline, metrics=metrics))
    gains = {}
    for baseline in baselines:
        gains[baseline] = {}
        for metric in METRICS:
            gain = float(
                np.mean([c["metrics"][metric]["gain"] for c in comparisons if c["baseline"] == baseline])
            )
            gains[baseline][metric] = gain
            if gain <= (1e-9 if metric == "log_esp" else 0):
                reasons.append(f"{baseline}: no strict suite-average improvement in {metric}")
    return dict(accepted=not reasons, reasons=reasons, comparisons=comparisons, mean_gains=gains)


def _validate(checkpoint, config, root, progress):
    config = validation_config(config)
    checkpoint = Path(checkpoint)
    source_data = checkpoint.read_bytes()
    source_digest = hashlib.sha256(source_data).hexdigest()
    payload = torch.load(io.BytesIO(source_data), map_location="cpu", weights_only=True)
    # Benchmark and promote the actual inference artifact, without Adam state or
    # the frozen teacher. last.pt retains those separately for training recovery.
    payload.pop("training_state", None)
    payload.setdefault("metadata", {})["checkpoint_role"] = "inference candidate; consult validation report"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    # Compile this immutable candidate, not a concurrently overwritten last.pt.
    candidate = root / "candidate.pt"
    torch.save(payload, candidate)
    data = candidate.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    meta = dict(
        config=config,
        checkpoint_sha256=digest,
        source_checkpoint_sha256=source_digest,
        environment=provenance(),
        policy="no per-case median regressions; strict mean improvement in all four metrics",
        scope="development selection; CPU full transpilation including policy loads",
        equivalence="exact terminal measurement projectors for all input states",
        research_validated=False,
    )
    meta["protocol_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows, cases = [], []
    fields = ["case", "algorithm", "seed", "status", "error", "equivalent", "esp", *METRICS, "fallback_steps"]
    with (root / "raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for n in config["sizes"]:
            for topology in config["topologies"]:
                device = synthetic(topology, n, config.get("device_seed", 177))
                device.save(root / f"{topology}-{n}.json")
                for family in config["families"]:
                    for circuit_seed in config["circuit_seeds"]:
                        case = f"{topology}-{n}-{family}-{circuit_seed}"
                        cases.append(case)
                        circuit = builtin(family, n, circuit_seed, config["circuit_depth"])
                        if circuit.count_ops().get("measure", 0):
                            raise ValueError("Validation generator must return unmeasured circuits")
                        circuit.measure_all()
                        with (root / f"{case}.qpy").open("wb") as qfile:
                            qpy.dump(circuit, qfile)
                        algorithms = [*config["baselines"], "learned"]
                        # Equal untimed warmups. Rotate measured order to reduce thermal/order bias.
                        for algorithm in algorithms:
                            try:
                                compile_one(circuit, device, algorithm, config["seeds"][0], candidate)
                            except Exception:
                                # Measured attempts record their errors below and fail the gate.
                                pass
                        for index, seed in enumerate(config["seeds"]):
                            order = (
                                algorithms[index % len(algorithms) :] + algorithms[: index % len(algorithms)]
                            )
                            for algorithm in order:
                                row = dict(case=case, algorithm=algorithm, seed=seed)
                                try:
                                    output, metrics = compile_one(circuit, device, algorithm, seed, candidate)
                                    check_equivalence(circuit, output, device)
                                    row.update(metrics, status="ok", equivalent=True, error="")
                                except Exception as exc:
                                    row.update(status="error", equivalent=False, error=str(exc))
                                writer.writerow(row)
                                stream.flush()
                                rows.append(row)
                            if progress and (index + 1) % 5 == 0:
                                progress(case=case, seeds_done=index + 1)
    report = assess(rows, cases, config["seeds"], config["baselines"])
    report.update(meta, rows=len(rows), cases=len(cases), seeds=len(config["seeds"]))
    # Recheck the frozen bytes before promotion, including changes during validation.
    if candidate.read_bytes() != data:
        report["accepted"] = False
        report["reasons"].append("candidate changed during validation")
    report["artifacts_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()
    }
    temporary_report = root / "report.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary_report.replace(root / "report.json")
    if report["accepted"]:
        temporary = root / "accepted.pt.tmp"
        temporary.write_bytes(data)
        temporary.replace(root / "accepted.pt")
    return report


def validate_checkpoint(checkpoint, config, output, progress=None):
    """Restore training RNGs, threads and routing environment after CPU validation."""
    config = validation_config(config)
    py_state, np_state, threads = random.getstate(), np.random.get_state(), torch.get_num_threads()
    try:
        with torch.random.fork_rng(), settings(NOISEROUTE_BEAM=1, NOISEROUTE_ENGINE=None):
            torch.set_num_threads(config["torch_threads"])
            return _validate(checkpoint, config, output, progress)
    finally:
        torch.set_num_threads(threads)
        random.setstate(py_state)
        np.random.set_state(np_state)
