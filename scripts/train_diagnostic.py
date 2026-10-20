"""Run a short guarded experiment in a fresh directory, without altering prior runs."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train-diagnostic.yaml")
    p.add_argument("--checkpoint", help="Optional existing weights; otherwise rebuild balanced imitation")
    p.add_argument("--out")
    args = p.parse_args()
    root = Path(args.out or ("runs/diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    if root.exists():
        p.error("Choose a new output directory")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        p.error("Checkpoint does not exist")
    root.mkdir(parents=True)
    print(f"Diagnostic output: {root.resolve()}", flush=True)

    def run(*arguments):
        subprocess.run([sys.executable, "-u", "-m", "noiseroute", *map(str, arguments)], check=True)

    checkpoint = args.checkpoint
    if not checkpoint:
        run("imitate", "--config", args.config, "--out", root / "bc")
        checkpoint = root / "bc/imitation.pt"
    run("train", "--config", args.config, "--checkpoint", checkpoint, "--out", root / "ppo")
    print(f"Checkpoint: {root / 'ppo/last.pt'}. Evaluate it before increasing the curriculum.", flush=True)


if __name__ == "__main__":
    main()
