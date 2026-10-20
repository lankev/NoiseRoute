"""Fresh imitation -> anchored PPO -> mandatory SABRE checkpoint selection."""

import argparse
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-competitive.yaml")
    parser.add_argument("--imitation", help="Reuse SABRE imitation weights (not PPO); keep their shape/model")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--out")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["torch_device"] = args.device
    if args.imitation:
        from noiseroute.model import load_model

        model, shape, metadata = load_model(args.imitation)
        if metadata.get("method") != "SABRE imitation":
            parser.error("--imitation requires a SABRE imitation checkpoint; PPO cannot be its own teacher")
        config.update(shape=asdict(shape), model=model.config)
        del model
    from noiseroute.acceptance import validation_config

    validation_config(config.get("validation", {}))
    root = Path(args.out or ("runs/competitive-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Competitive experiment: {root.resolve()}", flush=True)
    print("last.pt is for recovery. Only accepted.pt has passed the 20-seed gate.", flush=True)

    def run(*arguments):
        subprocess.run([sys.executable, "-u", "-m", "noiseroute", *map(str, arguments)], check=True)

    checkpoint = args.imitation
    if not checkpoint:
        run("imitate", "--config", config_path, "--out", root / "bc")
        checkpoint = root / "bc/imitation.pt"
    run("train", "--config", config_path, "--checkpoint", checkpoint, "--out", root / "ppo")
    accepted = root / "ppo/accepted.pt"
    print(
        f"Accepted: {accepted}"
        if accepted.exists()
        else f"No checkpoint accepted. Read {root / 'ppo/validation.json'} before another experiment.",
        flush=True,
    )


if __name__ == "__main__":
    main()
