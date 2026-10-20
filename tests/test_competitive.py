# ruff: noqa: E402
import copy
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit

torch = pytest.importorskip("torch")
from noiseroute import acceptance, objectives, train
from noiseroute.benchmark import compile_one
from noiseroute.device import synthetic
from noiseroute.env import RoutingEnv, Shape
from noiseroute.model import JointPolicy, load_model, save_model


def validation():
    return dict(
        sizes=[3],
        families=["random"],
        topologies=["line"],
        circuit_seeds=[200000],
        seeds=list(range(20)),
        circuit_depth=1,
        every_steps=4,
        baselines=["sabre"],
    )


def test_rates_are_independent_and_validate():
    config = dict(imitation_learning_rate=5e-5, ppo_learning_rate=1e-5, learning_rate=0.9)
    assert objectives.learning_rate(config, "imitation") == 5e-5
    assert objectives.learning_rate(config, "ppo") == 1e-5
    assert objectives.learning_rate({}, "imitation") == 5e-5
    assert objectives.learning_rate({}, "ppo") == 1e-5
    for rate in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            objectives.learning_rate(dict(ppo_learning_rate=rate), "ppo")


def test_masked_kl_zero_positive_and_gradient_to_actor_only():
    ref = torch.tensor([[1.0, -1.0, 90.0]], requires_grad=True)
    actor = torch.tensor([[-1.0, 1.0, -90.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])
    assert objectives.reference_kl(ref, ref, mask).item() == pytest.approx(0)
    loss = objectives.reference_kl(actor, ref, mask)
    assert loss > 0 and torch.isfinite(loss)
    loss.backward()
    assert ref.grad is None
    assert torch.isfinite(actor.grad).all() and actor.grad[0, 2] == 0
    before = actor.detach().clone()
    with torch.no_grad():
        actor -= 0.1 * actor.grad
    assert objectives.reference_kl(actor, ref, mask) < objectives.reference_kl(before, ref, mask)
    with pytest.raises(ValueError):
        objectives.reference_kl(actor, ref, torch.zeros_like(mask))


def test_comparative_reward_telescopes_and_failure_has_no_bonus(monkeypatch):
    calls = []

    def sabre(circuit, device, algorithm, seed):
        calls.append((algorithm, seed))
        return None, dict(log_esp=-0.2 - seed * 0.01)

    monkeypatch.setattr(objectives, "compile_one", sabre)
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.measure_all()
    env = RoutingEnv(circuit, synthetic("line", 3), joint=False, terminal_optimization_level=3)
    wrapper = objectives.SabreRelativeReward(env, list(range(20)))
    _, reward, done, _, info = wrapper.step(env.shape.nodes)
    assert done and info["completed"]
    assert reward == pytest.approx(env.terminal_log_esp - np.median([-0.2 - s * 0.01 for s in range(20)]))
    assert info["delta_log_esp"] == pytest.approx(reward)
    assert calls == [("sabre", s) for s in range(20)]
    calls.clear()
    failed = RoutingEnv(circuit, synthetic("line", 3), max_steps=1, terminal_optimization_level=3)
    wrapper = objectives.SabreRelativeReward(failed, list(range(20)))
    _, reward, done, _, info = wrapper.step(0)
    assert done and info["failed"] and reward == -10000 and calls == []


def passing_rows():
    return [
        dict(
            case="a",
            algorithm=alg,
            seed=s,
            status="ok",
            equivalent=True,
            fallback_steps=0,
            log_esp=-0.1 if alg == "learned" else -0.2,
            swaps_added=1 if alg == "learned" else 2,
            depth_2q=3 if alg == "learned" else 5,
            compile_ms=2.0 if alg == "learned" else 3.0,
        )
        for alg in ("sabre", "learned")
        for s in range(20)
    ]


def test_gate_rejects_missing_invalid_regressing_and_tied_results():
    rows = passing_rows()

    def judge(r):
        return acceptance.assess(r, ["a"], list(range(20)), ["sabre"])

    assert judge(rows)["accepted"]
    assert not judge(rows[:-1])["accepted"]
    assert not judge(rows + [rows[0]])["accepted"]
    for field, value in (
        ("status", "error"),
        ("equivalent", False),
        ("compile_ms", np.nan),
        ("swaps_added", None),
        ("fallback_steps", 1),
    ):
        bad = copy.deepcopy(rows)
        bad[-1][field] = value
        assert not judge(bad)["accepted"]
    for metric, value in (("log_esp", -0.3), ("swaps_added", 3), ("depth_2q", 7), ("compile_ms", 4)):
        bad = copy.deepcopy(rows)
        for row in bad:
            if row["algorithm"] == "learned":
                row[metric] = value
        assert not judge(bad)["accepted"]
    tied = copy.deepcopy(rows)
    for row in tied:
        row["swaps_added"] = 2
    assert not judge(tied)["accepted"]
    with pytest.raises(ValueError, match="20 distinct"):
        acceptance.assess(rows, ["a"], list(range(19)), ["sabre"])
    with pytest.raises(ValueError):
        acceptance.validation_config({**validation(), "seeds": [0] * 20})


def test_equivalence_accepts_layout_and_measurement_optimizations_rejects_wrong_result():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.rz(0.4, 2)
    circuit.measure_all()
    device = synthetic("line", 3)
    for algorithm in ("sabre", "noise_sabre"):
        routed, _ = compile_one(circuit, device, algorithm, 1)
        acceptance.check_equivalence(circuit, routed, device)
        # A prepended X changes the output projectors, even with identical wire counts.
        flip = QuantumCircuit(3)
        flip.x(1)
        wrong = flip.compose(circuit)
        with pytest.raises(ValueError, match="equivalence"):
            acceptance.check_equivalence(wrong, routed, device)


def test_zero_swaps_is_known_when_qiskit_skips_routing():
    from noiseroute.datasets import builtin

    circuit = builtin("ansatz", 5, 200000, 4)
    circuit.measure_all()
    device = synthetic("ring", 5, 177)
    _, metrics = compile_one(circuit, device, "noise_sabre", 0)
    assert metrics["swaps_added"] == 0


def test_frozen_reference_survives_ppo_and_resume(tmp_path, monkeypatch):
    config = dict(
        seed=4,
        shape=dict(nodes=8, edges=16, tokens=16),
        model=dict(dim=16, layers=1),
        curriculum=[dict(qubits=2, steps=8)],
        topologies=["line"],
        families=["ansatz"],
        circuit_depth=1,
        torch_threads=1,
        rollout_steps=4,
        ppo_epochs=1,
        batch_size=2,
        imitation_learning_rate=5e-5,
        ppo_learning_rate=1e-5,
        reference_kl_coef=0.05,
        validation=validation(),
    )
    teacher = JointPolicy(**config["model"])
    source = tmp_path / "imitation.pt"
    save_model(source, teacher, Shape(**config["shape"]), dict(method="SABRE imitation"))
    original = copy.deepcopy(teacher.state_dict())
    seen = []

    def reject(checkpoint, cfg, output, progress=None):
        seen.append(len(cfg["seeds"]))
        return dict(accepted=False)

    monkeypatch.setattr(acceptance, "validate_checkpoint", reject)
    report = train.Progress.report

    def interrupt(self, phase, force=False, **fields):
        report(self, phase, force, **fields)
        if phase == "saved" and fields["step"] == 4:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(train.Progress, "report", interrupt)
        with pytest.raises(KeyboardInterrupt):
            train.ppo(config, tmp_path / "first", source)
    source.unlink()  # Resume must use the embedded original reference, even if moved/deleted.
    result = train.ppo(config, tmp_path / "resume", tmp_path / "first/last.pt", resume=True)
    payload = torch.load(result, weights_only=True)
    for name, value in original.items():
        assert torch.equal(payload["training_state"]["imitation_reference"]["state_dict"][name], value)
    assert any(not torch.equal(payload["state_dict"][k], original[k]) for k in original)
    assert payload["training_state"]["optimizer"]["param_groups"][0]["lr"] == 1e-5
    assert seen == [20] and not (tmp_path / "resume/accepted.pt").exists()
    assert load_model(result)[2]["imitation_reference_sha256"]
    updates = json.loads((tmp_path / "resume/train.json").read_text())["updates"]
    assert all(np.isfinite(u["imitation_kl"]) and u["imitation_kl"] >= -1e-7 for u in updates)


def test_real_twenty_seed_validation_rejects_untrained_policy_and_restores_rng(tmp_path):
    source = tmp_path / "untrained.pt"
    save_model(source, JointPolicy(dim=16, layers=1), Shape(8, 16, 16))
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    report = acceptance.validate_checkpoint(source, validation(), tmp_path / "validation")
    assert report["seeds"] == 20 and report["rows"] == 40
    assert torch.equal(state, torch.get_rng_state()) and torch.get_num_threads() == threads
    # This test checks actual compilation/equivalence, not timing-dependent superiority.
    assert report["cases"] == 1
    rows = list(__import__("csv").DictReader((tmp_path / "validation/raw.csv").open()))
    assert all(r["status"] == "ok" for r in rows), rows
    assert (tmp_path / "validation/accepted.pt").exists() == report["accepted"]


@pytest.mark.parametrize("mode", ["pass", "regression", "tamper", "report_error"])
def test_promotion_uses_only_validated_inference_weights(tmp_path, monkeypatch, mode):
    """Synthetic metrics exercise promotion, not a claimed win over SABRE."""
    from pathlib import Path

    source = tmp_path / "source.pt"
    save_model(
        source, JointPolicy(dim=16, layers=1), Shape(8, 16, 16), training_state=dict(optimizer={"test": True})
    )
    original = source.read_bytes()
    output = tmp_path / "validation"

    def compile_mock(circuit, device, algorithm, seed, checkpoint):
        good = algorithm == "learned"
        return circuit.copy(), dict(
            log_esp=-0.1 if good else -0.2,
            esp=0.9 if good else 0.8,
            swaps_added=1 if good else 2,
            depth_2q=2 if good else 3,
            compile_ms=(4 if mode == "regression" else 1) if good else 2,
            fallback_steps=0,
        )

    def progress(**fields):
        if mode == "tamper" and fields["seeds_done"] == 20:
            with (output / "candidate.pt").open("ab") as stream:
                stream.write(b"changed")

    replace = Path.replace

    def guarded_replace(path, target):
        if Path(target).name == "report.json" and mode == "report_error":
            raise OSError("simulated report write failure")
        if Path(target).name == "accepted.pt":
            assert (output / "report.json").is_file()
        return replace(path, target)

    monkeypatch.setattr(acceptance, "compile_one", compile_mock)
    monkeypatch.setattr(Path, "replace", guarded_replace)
    config = {**validation(), "sizes": [2]}
    if mode == "report_error":
        with pytest.raises(OSError, match="simulated"):
            acceptance.validate_checkpoint(source, config, output, progress)
    else:
        report = acceptance.validate_checkpoint(source, config, output, progress)
        assert report["accepted"] == (mode == "pass")
    assert (output / "accepted.pt").exists() == (mode == "pass")
    assert source.read_bytes() == original
    if mode == "pass":
        payload = torch.load(output / "accepted.pt", weights_only=True)
        assert "training_state" not in payload
        assert (output / "accepted.pt").read_bytes() == (output / "candidate.pt").read_bytes()
