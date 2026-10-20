# ruff: noqa: E402
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
from noiseroute import model as model_module
from noiseroute import train
from noiseroute.env import Shape
from noiseroute.model import JointPolicy, load_model, save_model


@pytest.fixture
def config():
    return dict(
        seed=4,
        shape=dict(nodes=8, edges=16, tokens=16),
        model=dict(dim=16, layers=1),
        curriculum=[dict(qubits=3, steps=8), dict(qubits=5, steps=8)],
        topologies=["line"],
        families=["random"],
        circuit_depth=2,
        torch_threads=1,
        rollout_steps=4,
        ppo_epochs=1,
        batch_size=2,
        progress_interval_seconds=0,
    )


@pytest.mark.parametrize("stop", [4, 8])
def test_resume_new_checkpoint_keeps_steps_and_adam(config, tmp_path, monkeypatch, stop):
    original_report = train.Progress.report

    def interrupt(progress, phase, force=False, **fields):
        original_report(progress, phase, force=force, **fields)
        if phase == "saved" and fields["saved_step"] == stop:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(train.Progress, "report", interrupt)
        with pytest.raises(KeyboardInterrupt):
            train.ppo(config, tmp_path / "first")
    checkpoint = tmp_path / "first/last.pt"
    assert load_model(checkpoint)[2]["steps"] == stop
    status = json.loads((tmp_path / "first/status.json").read_text())
    assert status["phase"] == "interrupted" and status["saved_step"] == stop
    before = checkpoint.read_bytes()
    resumed = train.ppo(config, tmp_path / "resumed", checkpoint, resume=True)
    assert checkpoint.read_bytes() == before
    payload = torch.load(resumed, weights_only=True)
    assert payload["metadata"]["steps"] == 16
    # 4 rollouts * 2 minibatches, including the work before the interruption.
    assert max(float(s["step"]) for s in payload["training_state"]["optimizer"]["state"].values()) == 8
    log = json.loads((tmp_path / "resumed/train.json").read_text())
    assert log["starting_step"] == stop
    assert [u["step"] for u in log["updates"]] == list(range(stop + 4, 17, 4))
    assert log["updates"][0]["stage"] == (1 if stop == 8 else 0)


def test_legacy_checkpoint_and_windows_space_paths(config, tmp_path):
    previous = tmp_path / "NoiseRoute (2)" / "old run"
    checkpoint = previous / "last.pt"
    save_model(
        checkpoint,
        JointPolicy(**config["model"]),
        Shape(**config["shape"]),
        dict(method="PPO", steps=8, config=config),
    )
    config_path = tmp_path / "training config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "new run"
    script = Path(__file__).resolve().parents[1] / "scripts/resume_training.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            str(previous),
            "--config",
            str(config_path),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Adam starts fresh" in result.stdout
    assert "phase=optimize" in result.stdout and "phase=finished" in result.stdout
    assert load_model(output / "last.pt")[2]["steps"] == 16
    assert not (output / "stage_0.pt").exists()


def test_recovery_falls_back_to_imitation(config, tmp_path):
    checkpoint = tmp_path / "bc/imitation.pt"
    save_model(
        checkpoint,
        JointPolicy(**config["model"]),
        Shape(**config["shape"]),
        dict(method="SABRE imitation", config=config),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = Path(__file__).resolve().parents[1] / "scripts/resume_training.py"
    output = tmp_path / "new"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            str(tmp_path / "missing"),
            "--imitation",
            str(checkpoint),
            "--config",
            str(config_path),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "start PPO from completed imitation" in result.stdout
    assert load_model(output / "last.pt")[2]["steps"] == 16


def test_interrupted_checkpoint_write_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    model = JointPolicy(dim=16, layers=1)
    save_model(path, model, Shape(8, 16, 16), dict(steps=4))
    before = path.read_bytes()

    def interrupted_save(payload, target):
        Path(target).write_bytes(b"incomplete")
        raise KeyboardInterrupt

    monkeypatch.setattr(model_module.torch, "save", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        save_model(path, model, Shape(8, 16, 16), dict(steps=8))
    assert path.read_bytes() == before
    assert load_model(path)[2]["steps"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupt_during_optimizer_keeps_last_complete_update(config, tmp_path, monkeypatch):
    original_step = torch.optim.Adam.step
    calls = 0

    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        if calls == 3:  # First rollout saved; second rollout partly updated.
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train.ppo(config, tmp_path / "interrupted")
    payload = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    assert payload["metadata"]["steps"] == 4
    assert max(float(s["step"]) for s in payload["training_state"]["optimizer"]["state"].values()) == 2
    status = json.loads((tmp_path / "interrupted/status.json").read_text())
    assert status["step"] == 8 and status["saved_step"] == 4


def test_resume_guards_and_one_transfer_per_rollout(config, tmp_path, monkeypatch):
    calls = []
    original_batch = train.batch_observations

    def count_batch(samples, device):
        calls.append(len(samples))
        return original_batch(samples, device)

    monkeypatch.setattr(train, "batch_observations", count_batch)
    checkpoint = train.ppo(config, tmp_path / "first")
    assert calls == [4, 4, 4, 4]
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        train.ppo(config, tmp_path / "first", checkpoint, resume=True)
    assert checkpoint.read_bytes() == before
    changed = dict(config, curriculum=[dict(qubits=3, steps=32)])
    with pytest.raises(ValueError, match="same training configuration"):
        train.ppo(changed, tmp_path / "bad", checkpoint, resume=True)
    with pytest.raises(ValueError, match="requires a PPO"):
        train.ppo(config, tmp_path / "missing", resume=True)
    changed = dict(config, shape=dict(nodes=9, edges=16, tokens=16))
    with pytest.raises(ValueError, match="shape differs"):
        train.ppo(changed, tmp_path / "shape", checkpoint)
