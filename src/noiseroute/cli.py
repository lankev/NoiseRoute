import argparse
import json
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description="NoiseRoute research pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--out")
    data = sub.add_parser("data")
    data.add_argument("--out", default="data/builtin")
    data.add_argument("--sizes", nargs="+", type=int, default=[5, 10])
    data.add_argument("--seeds", nargs="+", type=int, default=[100000])
    data.add_argument("--families", nargs="*", default=["qft", "grover", "ansatz", "random", "arithmetic"])
    data.add_argument("--mqt", action="store_true")
    data.add_argument("--qasmbench")
    data.add_argument("--max-qubits", type=int, default=20)
    data.add_argument("--split", choices=["train", "validation", "test"], default="test")
    snap = sub.add_parser("device")
    snap.add_argument("--kind", choices=["ring", "line", "grid", "neutral", "heavy_hex"], default="ring")
    snap.add_argument("--n", type=int, default=5)
    snap.add_argument("--seed", type=int, default=0)
    snap.add_argument("--fake")
    snap.add_argument("--out", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--config", default="configs/bench-smoke.yaml")
    bench.add_argument("--out", required=True)
    acceptance = sub.add_parser("accept-checkpoint", help="Validate >=20 seeds before any promotion")
    acceptance.add_argument("--checkpoint", required=True)
    acceptance.add_argument("--config", default="configs/train-competitive.yaml")
    acceptance.add_argument("--out", required=True)
    for command in ("imitate", "train", "sb3", "distill"):
        s = sub.add_parser(command)
        s.add_argument("--config", default="configs/train-laptop.yaml")
        s.add_argument("--out", required=True)
        if command in ("train", "distill"):
            s.add_argument("--checkpoint", required=command == "distill")
        if command == "train":
            s.add_argument(
                "--resume", action="store_true", help="Continue saved PPO steps; requires --checkpoint"
            )
        if command == "distill":
            s.add_argument("--episodes", type=int, default=16)
    ex = sub.add_parser("export")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--out", required=True)
    lat = sub.add_parser("latency")
    lat.add_argument("--model", required=True)
    lat.add_argument("--engine", choices=["onnx", "openvino"], default="onnx")
    lat.add_argument("--device", choices=["CPU", "NPU", "GPU"], default="CPU")
    lat.add_argument("--iterations", type=int, default=200)
    lat.add_argument("--out")
    route = sub.add_parser("route")
    route.add_argument("--input", required=True)
    route.add_argument("--device", required=True)
    route.add_argument("--out", required=True)
    route.add_argument(
        "--algorithm", choices=["sabre", "noise_sabre", "noiseroute", "learned", "ai"], default="noiseroute"
    )
    route.add_argument("--checkpoint")
    route.add_argument("--seed", type=int, default=0)
    val = sub.add_parser("validate-noise")
    val.add_argument("--device", required=True)
    val.add_argument("--n", type=int, default=5)
    val.add_argument("--family", default="random")
    val.add_argument("--shots", type=int, default=8192)
    val.add_argument("--out", required=True)
    val.add_argument("--checkpoint")
    args = p.parse_args()
    if args.command == "doctor":
        from .benchmark import provenance
        from .core import core_class

        result = provenance()
        result["routing_core"] = core_class().__name__
        try:
            import torch

            result["cuda_available"] = torch.cuda.is_available()
            result["cuda_devices"] = [
                dict(
                    name=torch.cuda.get_device_name(i),
                    memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                )
                for i in range(torch.cuda.device_count())
            ]
        except ImportError:
            result["torch"] = "not installed"
        try:
            import openvino as ov

            result["openvino_devices"] = ov.Core().available_devices
        except ImportError:
            result["openvino"] = "not installed"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2))
    elif args.command == "data":
        from .datasets import prepare

        records, errors = prepare(
            args.out,
            args.sizes,
            args.seeds,
            args.families,
            args.mqt,
            args.qasmbench,
            args.max_qubits,
            args.split,
        )
        result = dict(circuits=len(records), errors=errors, manifest=f"{args.out}/manifest.json")
    elif args.command == "device":
        from .device import Device, synthetic

        if args.fake:
            from qiskit_ibm_runtime import fake_provider

            if not args.fake.startswith("Fake") or not hasattr(fake_provider, args.fake):
                raise ValueError("Unknown FakeBackend class")
            device = Device.from_target(getattr(fake_provider, args.fake)().target, args.fake)
        else:
            device = synthetic(args.kind, args.n, args.seed)
        device.save(args.out)
        result = dict(device=device.name, qubits=device.n, edges=len(device.edges), source=device.source)
    elif args.command == "bench":
        from .benchmark import run

        rows = run(yaml.safe_load(Path(args.config).read_text()), args.out)
        result = dict(rows=len(rows), errors=sum(r["status"] != "ok" for r in rows), output=args.out)
    elif args.command == "accept-checkpoint":
        from .acceptance import validate_checkpoint

        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        report = validate_checkpoint(args.checkpoint, config.get("validation", config), args.out)
        result = dict(
            accepted=report["accepted"], reasons=report["reasons"], report=f"{args.out}/report.json"
        )
    elif args.command in ("imitate", "train", "sb3", "distill"):
        from .train import distill, imitation, ppo, sb3_baseline

        config = yaml.safe_load(Path(args.config).read_text())
        if args.command == "imitate":
            output = imitation(config, args.out)
        elif args.command == "train":
            output = ppo(config, args.out, args.checkpoint, resume=args.resume)
        elif args.command == "sb3":
            sb3_baseline(config, args.out)
            output = args.out
        else:
            output = distill(config, args.checkpoint, args.out, args.episodes)
        result = dict(output=str(output))
        if args.command == "train":
            accepted = Path(args.out) / "accepted.pt"
            result.update(
                recovery_checkpoint=str(output),
                accepted_checkpoint=str(accepted) if accepted.exists() else None,
            )
    elif args.command == "export":
        from .export import export_onnx

        result = export_onnx(args.checkpoint, args.out)
    elif args.command == "latency":
        from .export import latency

        result = latency(args.model, args.engine, args.device, args.iterations, args.out)
    elif args.command == "route":
        from qiskit import qasm2, qpy

        from .benchmark import compile_one
        from .device import Device

        path = Path(args.out)
        if path.suffix != ".qpy":
            raise ValueError("Output must be .qpy; QASM loses layout metadata")
        qc = qasm2.load(args.input, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        routed, result = compile_one(qc, Device.load(args.device), args.algorithm, args.seed, args.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            qpy.dump(routed, stream)
        path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    elif args.command == "validate-noise":
        from .benchmark import compile_one
        from .datasets import builtin
        from .device import Device
        from .metrics import noisy_validation

        device = Device.load(args.device)
        qc = builtin(args.family, args.n, 100000, 3)
        measured = qc.copy()
        measured.measure_all()
        result = {}
        for algorithm in ["sabre", "noise_sabre", "learned" if args.checkpoint else "noiseroute"]:
            compiled, metrics = compile_one(measured, device, algorithm, 0, args.checkpoint)
            result[algorithm] = dict(
                metrics=metrics, validation=noisy_validation(qc, compiled, device, args.shots)
            )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, default=str))
    if args.command == "accept-checkpoint" and not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
