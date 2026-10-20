# ruff: noqa: E402
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from noiseroute import train
from noiseroute.model import load_model


@pytest.fixture
def config():
    return dict(
        seed=41,
        shape=dict(nodes=8, edges=16, tokens=16),
        model=dict(dim=16, layers=1),
        curriculum=[dict(qubits=2, steps=8), dict(qubits=2, steps=8)],
        topologies=["line"],
        families=["ansatz"],
        circuit_depth=1,
        torch_threads=1,
        rollout_steps=4,
        ppo_epochs=2,
        batch_size=2,
        reward_scale=0.01,
        value_loss="huber",
        target_kl=0.02,
        quality_gate=dict(window=4, min_episodes=2, max_failure_rate=0.25, min_success_rate=0.9),
    )


@pytest.mark.parametrize("topologies", [["line", "ring"], ["line", "ring", "grid"]])
def test_demonstrations_cover_independent_dimensions(topologies):
    config = dict(
        curriculum=[dict(qubits=n) for n in (5, 20, 50)], topologies=topologies, families=["random", "ansatz"]
    )
    count = 3 * len(topologies) * 2
    selected = {
        (stage["qubits"], top, fam)
        for stage, top, fam in [train.demonstration_case(config, ep) for ep in range(count)]
    }
    assert selected == {(n, t, f) for n in (5, 20, 50) for t in topologies for f in ("random", "ansatz")}


def test_empty_demo_coverage_is_reported_before_training(config, tmp_path, monkeypatch):
    config.update(require_demo_coverage=True, imitation_episodes=2)
    monkeypatch.setattr(train, "expert_trace", lambda *args: [])
    with pytest.raises(ValueError, match="Missing complete demonstrations"):
        train.imitation(config, tmp_path)
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage and all(row["accepted"] == 0 for row in coverage.values())
    assert not (tmp_path / "imitation.pt").exists()


def test_positive_scale_preserves_failure_penalty_order_and_huber_gradient():
    raw = [-0.03, -0.1, -10000.0]
    scaled = train.training_rewards(raw, 0.01)
    np.testing.assert_allclose(scaled.sum(), sum(raw) * 0.01, rtol=1e-6)
    assert scaled[-1] < scaled[1] < scaled[0]
    for scale in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            train.training_rewards(raw, scale)
    value = torch.tensor([0.0], requires_grad=True)
    loss = train.value_loss(value, torch.tensor([-100.0]), "huber")
    loss.backward()
    assert torch.isfinite(loss) and abs(value.grad.item()) <= 1


def test_failed_rollout_stops_before_next_stage_and_keeps_checkpoint(config, tmp_path, monkeypatch):
    make_env = train.make_env

    def failing_env(*args, **kwargs):
        env = make_env(*args, **kwargs)
        env.max_steps = 1  # Placement cannot finish: real environment's -10000 failure.
        return env

    monkeypatch.setattr(train, "make_env", failing_env)
    with pytest.raises(train.TrainingQualityError, match="Quality gate stopped"):
        train.ppo(config, tmp_path / "run")
    ck = tmp_path / "run/last.pt"
    meta = load_model(ck)[2]
    assert meta["steps"] == 4 and meta["training_quality"]["collapse"]
    assert not (tmp_path / "run/stage_1.pt").exists()
    update = json.loads((tmp_path / "run/train.json").read_text())["updates"][0]
    assert update["failed"] == 4 and update["completed"] == 0
    assert update["mean_reward"] <= -10000 and -101 < update["mean_training_reward"] <= -100
    assert 0 <= update["value_loss"] < 101
    assert json.loads((tmp_path / "run/status.json").read_text())["phase"] == "quality_stop"
    with pytest.raises(train.TrainingQualityError, match="did not pass"):
        train.ppo(config, tmp_path / "resume", ck, resume=True)


def test_gate_allows_successful_stages_and_logs_loss_components(config, tmp_path):
    ck = train.ppo(config, tmp_path / "run")
    assert load_model(ck)[2]["training_quality"]["status"] == "pass"
    log = json.loads((tmp_path / "run/train.json").read_text())
    assert {u["stage"] for u in log["updates"]} == {0, 1}
    for u in log["updates"]:
        assert all(
            np.isfinite(u[k]) for k in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        )
        assert u["optimizer_steps"] > 0
        assert u["mean_training_reward"] == pytest.approx(u["mean_reward"] * 0.01, rel=1e-5)
    resumed = train.ppo(config, tmp_path / "resume", tmp_path / "run/stage_0.pt", resume=True)
    assert load_model(resumed)[2]["steps"] == 16


def test_gate_requires_enough_episodes():
    gate = dict(window=10, min_episodes=5, min_success_rate=0.9, max_failure_rate=0.25)
    assert train.quality_status([1], gate)["status"] == "insufficient"
    assert train.quality_status([1] * 10, gate)["status"] == "pass"
    assert train.quality_status([1] * 2 + [0] * 8, gate)["collapse"]
