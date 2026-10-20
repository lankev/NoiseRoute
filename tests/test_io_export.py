import hashlib
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from noiseroute.benchmark import compile_one, run
from noiseroute.datasets import builtin, load_manifest, prepare
from noiseroute.device import Device, synthetic
from noiseroute.env import RoutingEnv, Shape


@pytest.mark.parametrize("native_newline", ["\n", "\r\n"], ids=["lf", "windows-crlf"])
def test_dataset_roundtrip_and_tamper(tmp_path, monkeypatch, native_newline):
    original_open = Path.open

    def platform_open(path, mode="r", *args, **kwargs):
        # Reproduce Windows text-mode writes even when CI itself runs on Linux.
        if "w" in mode and "b" not in mode and kwargs.get("newline") is None:
            kwargs["newline"] = native_newline
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", platform_open)
    rows, errors = prepare(tmp_path, sizes=[3], seeds=[10], families=["random"])
    assert not errors
    records = load_manifest(tmp_path / "manifest.json", split="test")
    assert len(records) == 1
    payload = (tmp_path / rows[0]["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == rows[0]["sha256"]
    assert b"\r\n" not in payload
    assert Operator(records[0][1]).equiv(Operator(builtin("random", 3, 10)))
    (tmp_path / rows[0]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(tmp_path / "manifest.json")


def test_minimum_seeds(tmp_path):
    with pytest.raises(ValueError, match="20 seeds"):
        run(dict(seeds=[0], devices=[]), tmp_path / "run")


def test_distinct_seeds(tmp_path):
    with pytest.raises(ValueError, match="20 seeds"):
        run(dict(seeds=[0] * 20, devices=[]), tmp_path / "run")


def test_snapshot_roundtrip(tmp_path):
    d = synthetic("grid", 6, 3)
    d.save(tmp_path / "device.json")
    loaded = Device.load(tmp_path / "device.json")
    assert loaded.instructions == d.instructions
    np.testing.assert_array_equal(loaded.dist, d.dist)


def test_onnx_and_plugin(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from noiseroute.export import RuntimePolicy, export_onnx
    from noiseroute.model import JointPolicy, save_model

    torch.set_num_threads(1)
    save_model(tmp_path / "random.pt", JointPolicy(dim=16, layers=1), Shape(8, 16, 16))
    report = export_onnx(tmp_path / "random.pt", tmp_path / "policy.onnx")
    assert report["max_abs_error"] < 2e-4
    policy = RuntimePolicy(tmp_path / "policy.onnx")
    q, d = builtin("random", 3, 2, 2), synthetic("line", 3)
    env = RoutingEnv(q, d, shape=policy.shape)
    assert env.action_masks()[policy(env.observe()).argmax()]
    monkeypatch.setenv("NOISEROUTE_ENGINE", "onnx")
    out, metrics = compile_one(q, d, "learned", checkpoint=str(tmp_path / "policy.onnx"))
    assert Operator.from_circuit(out).equiv(Operator(q))
    assert metrics["fallback_steps"] >= 0


def test_noisy_simulation():
    pytest.importorskip("qiskit_aer")
    from noiseroute.metrics import noisy_validation

    q, d = builtin("random", 3, 8, 1), synthetic("line", 3)
    measured = q.copy()
    measured.measure_all()
    out, _ = compile_one(measured, d, "noiseroute")
    result = noisy_validation(q, out, d, shots=256)
    assert 0 <= result["hellinger"] <= 1.000001
