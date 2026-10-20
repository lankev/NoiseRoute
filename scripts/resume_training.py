"""Continue a PPO run in a new folder, or reuse its completed imitation."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Previous PPO directory")
    parser.add_argument("--imitation", help="Fallback checkpoint if PPO never saved")
    parser.add_argument("--config", default="configs/train-laptop.yaml")
    parser.add_argument("--out")
    args = parser.parse_args()
    checkpoint = Path(args.run) / "last.pt"
    resume = checkpoint.is_file()
    if not resume:
        stages = list(Path(args.run).glob("stage_*.pt"))
        if stages:
            checkpoint = max(stages, key=lambda p: int(p.stem.split("_")[-1]))
            resume = True
    if not resume:
        if not args.imitation or not Path(args.imitation).is_file():
            parser.error("No PPO checkpoint found. Supply the completed --imitation checkpoint.")
        checkpoint = Path(args.imitation)
    output = args.out or str(Path("runs") / ("ppo-resume-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    print(f"Checkpoint: {checkpoint.resolve()}", flush=True)
    print(f"Output: {Path(output).resolve()}", flush=True)
    print(
        "Mode: resume saved PPO steps" if resume else "Mode: start PPO from completed imitation", flush=True
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "noiseroute",
        "train",
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--out",
        output,
    ]
    if resume:
        command.append("--resume")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
