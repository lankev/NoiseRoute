"""Recompute the reported small benchmark and a complete short ML pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(args.out).resolve()
if out.exists():
    raise SystemExit("Choose a new output directory to preserve earlier measurements.")
out.mkdir(parents=True)


def run(*command):
    subprocess.run([sys.executable, "-m", "noiseroute", *map(str, command)], cwd=root, check=True)


run("doctor", "--out", out / "doctor.json")
run(
    "bench",
    "--config",
    "configs/bench-smoke.yaml" if args.smoke else "configs/bench-small20.yaml",
    "--out",
    out / "bench",
)
run("imitate", "--config", "configs/train-smoke.yaml", "--out", out / "bc")
run(
    "train",
    "--config",
    "configs/train-smoke.yaml",
    "--checkpoint",
    out / "bc/imitation.pt",
    "--out",
    out / "ppo",
)
run("export", "--checkpoint", out / "ppo/last.pt", "--out", out / "policy.onnx")
run("latency", "--model", out / "policy.onnx", "--out", out / "latency-onnx.json")
run(
    "latency",
    "--model",
    out / "policy.onnx",
    "--engine",
    "openvino",
    "--device",
    "CPU",
    "--out",
    out / "latency-openvino.json",
)
run(
    "validate-noise", "--device", out / "bench/line_5_s7.json", "--shots", "1024", "--out", out / "noise.json"
)
print(f"Results: {out}. Timings and sampled outcomes may differ from the bundled run.")
