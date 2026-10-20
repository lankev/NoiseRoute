import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from qiskit import qpy, transpile

from .circuit import normalize
from .datasets import builtin, load_manifest
from .device import Device, synthetic
from .metrics import evaluate
from .router import noise_layout


@contextmanager
def settings(**values):
    previous = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def compile_one(qc, device, algorithm, seed=0, checkpoint=None):
    swaps, fallback, seen = 0, 0, False

    def callback(**kw):
        nonlocal swaps, fallback, seen
        # A successful CheckMap can skip every routing pass (zero added SWAPs).
        # Do not report an unknown count in that explicitly certified case.
        if kw["pass_"].__class__.__name__ == "CheckMap" and (
            kw["property_set"].get("is_swap_mapped") or kw["property_set"].get("routing_not_needed")
        ):
            seen = True
        if kw["pass_"].__class__.__name__ in ("SabreLayout", "SabreSwap", "AIRouting", "NoiseRouting"):
            swaps = max(swaps, kw["dag"].count_ops().get("swap", 0))
            seen = True
        info = kw["property_set"].get("noiseroute_stats")
        if info:
            swaps, fallback = info["swaps"], info["fallback_steps"]

    if algorithm == "learned" and not checkpoint:
        raise ValueError("learned requires an explicit checkpoint")
    start = time.perf_counter()
    qc = normalize(qc)
    options = dict(target=device.target(), seed_transpiler=seed, optimization_level=3, callback=callback)
    with settings(NOISEROUTE_MODEL=checkpoint if algorithm == "learned" else None, NOISEROUTE_DEVICE=None):
        if algorithm in ("sabre", "noise_sabre"):
            options.update(layout_method="sabre", routing_method="sabre")
            if algorithm == "noise_sabre":
                options.update(initial_layout=noise_layout(qc, device, seed), layout_method="trivial")
            output = transpile(qc, **options)
        elif algorithm in ("noiseroute", "learned"):
            output = transpile(qc, layout_method="noiseroute", routing_method="noiseroute", **options)
        elif algorithm == "ai":
            import subprocess
            import tempfile

            from qiskit import qasm2

            python = os.environ.get("NOISEROUTE_AI_PYTHON")
            if not python:
                raise RuntimeError("Set NOISEROUTE_AI_PYTHON to the isolated worker Python; docs/INSTALL.md")
            with tempfile.TemporaryDirectory(prefix="noiseroute-ai-") as temp:
                temp = Path(temp)
                qasm2.dump(qc, temp / "input.qasm")
                device.save(temp / "device.json")
                completed = subprocess.run(
                    [python, "-m", "noiseroute.ai_worker", str(temp), str(seed)],
                    capture_output=True,
                    text=True,
                    timeout=float(os.environ.get("NOISEROUTE_AI_TIMEOUT", "120")),
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(f"AIRouting worker failed: {completed.stderr[-2000:]}")
                with (temp / "output.qpy").open("rb") as stream:
                    output = qpy.load(stream)[0]
                worker = json.loads((temp / "metrics.json").read_text())
            worker["compile_ms_inner"] = worker["compile_ms"]
            worker["compile_ms"] = (time.perf_counter() - start) * 1000
            worker["seed_controls_router"] = False
            return output, worker
        else:
            raise ValueError(algorithm)
    elapsed = (time.perf_counter() - start) * 1000
    result = evaluate(output, device)
    result.update(
        compile_ms=elapsed,
        swaps_added=swaps if seen else None,
        fallback_steps=fallback,
        seed_controls_router=algorithm != "ai",
    )
    return output, result


def provenance():
    versions = {}
    for name in (
        "noiseroute",
        "qiskit",
        "qiskit-aer",
        "torch",
        "numpy",
        "gymnasium",
        "noiseroute-core",
        "mqt.bench",
        "openvino",
        "onnxruntime",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(
        platform=platform.platform(),
        processor=platform.processor(),
        python=platform.python_version(),
        packages=versions,
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS")},
    )


def run(config, out):
    out = Path(out)
    if (out / "raw.csv").exists():
        raise FileExistsError(f"{out}/raw.csv exists; use a new directory")
    seeds = config.get("seeds", list(range(20)))
    if len(set(seeds)) < 20 and not config.get("smoke", False):
        raise ValueError("Research protocol requires >=20 seeds; smoke: true permits a quick check")
    out.mkdir(parents=True, exist_ok=True)
    if config.get("manifest"):
        circuits = load_manifest(config["manifest"], config.get("split", "test"))
    else:
        circuits = [
            (dict(path=f"{f}_{n}_s{s}", family=f, source="builtin", seed=s), builtin(f, n, s, 3))
            for f in config.get("families", ["random", "qft"])
            for n in config.get("sizes", [5])
            for s in config.get("circuit_seeds", [100000])
        ]
    devices = [
        Device.load(d["snapshot"]) if "snapshot" in d else synthetic(d["kind"], d["n"], d.get("seed", 0))
        for d in config["devices"]
    ]
    checkpoint = config.get("checkpoint")
    meta = dict(
        config=config,
        environment=provenance(),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
    )
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    fields = [
        "circuit",
        "family",
        "source",
        "device",
        "algorithm",
        "seed",
        "status",
        "error",
        "esp",
        "log_esp",
        "log_gate",
        "log_idle",
        "depth_2q",
        "native_2q",
        "duration_s",
        "idle_s",
        "compile_ms",
        "swaps_added",
        "fallback_steps",
        "seed_controls_router",
        "compile_ms_inner",
        "ai_qiskit",
        "ai_version",
    ]
    rows = []
    with (out / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dev in devices:
            dev.save(out / f"{dev.name}.json")
            for record, qc in circuits:
                if qc.num_qubits > dev.n:
                    continue
                if config.get("measure_all", True) and not qc.count_ops().get("measure", 0):
                    qc = qc.copy()
                    qc.measure_all()
                name = Path(record["path"]).stem
                with (out / f"{name}.qpy").open("wb") as stream:
                    qpy.dump(qc, stream)
                for algorithm in config["algorithms"]:
                    for seed in seeds:
                        row = dict(
                            circuit=name,
                            family=record["family"],
                            source=record["source"],
                            device=dev.name,
                            algorithm=algorithm,
                            seed=seed,
                        )
                        try:
                            _, metrics = compile_one(qc, dev, algorithm, seed, checkpoint)
                            row.update(status="ok", error="", **metrics)
                        except Exception as exc:
                            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                        writer.writerow(row)
                        f.flush()
                        rows.append(row)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.suffix in (".qpy", ".json", ".csv")
    }
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2))
    summarize(out / "raw.csv", out / "summary.json")
    return rows


def summarize(csv_path, output):
    with open(csv_path) as stream:
        rows = list(csv.DictReader(stream))
    algorithms = sorted({r["algorithm"] for r in rows})
    groups, medians, paired = {}, {}, {}
    for a in algorithms:
        valid = [r for r in rows if r["algorithm"] == a and r["status"] == "ok"]
        summary = dict(ok=len(valid), errors=sum(r["algorithm"] == a and r["status"] != "ok" for r in rows))
        for metric in ("log_esp", "swaps_added", "compile_ms", "depth_2q"):
            vals = [float(r[metric]) for r in valid if r[metric] != ""]
            if vals:
                summary[metric] = dict(median=float(np.median(vals)), p95=float(np.percentile(vals, 95)))
        groups[a] = summary
    for r in rows:
        if r["status"] == "ok":
            medians.setdefault((r["circuit"], r["device"], r["algorithm"]), []).append(float(r["log_esp"]))
    for a in algorithms:
        if a == "noise_sabre":
            continue
        differences = [
            float(np.median(v) - np.median(medians[(c, d, "noise_sabre")]))
            for (c, d, alg), v in medians.items()
            if alg == a and (c, d, "noise_sabre") in medians
        ]
        if differences:
            arr = np.array(differences)
            rng = np.random.default_rng(0)
            boots = np.median(rng.choice(arr, (2000, len(arr))), axis=1)
            paired[a] = dict(
                paired_circuit_devices=len(arr),
                win_fraction=float(np.mean(arr > 1e-10)),
                loss_fraction=float(np.mean(arr < -1e-10)),
                median_delta_log_esp=float(np.median(arr)),
                bootstrap_95_ci=np.percentile(boots, [2.5, 97.5]).tolist(),
            )
    result = dict(
        algorithms=groups,
        paired_vs_noise_sabre=paired,
        note="Exploratory bootstrap; correlated family/device effects require stratified analysis.",
    )
    Path(output).write_text(json.dumps(result, indent=2))
    return result
