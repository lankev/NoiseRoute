# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from noiseroute.datasets import builtin
from noiseroute.device import synthetic
from noiseroute.env import RoutingEnv, Shape
from noiseroute.model import JointPolicy, tensor_obs
from noiseroute.train import expert_trace, gae, imitation, ppo


def test_gae_terminal_no_leak():
    _, ret = gae([1, 2, 100], [0, 0, 0], [False, True, True], 900, gamma=1, lam=1)
    np.testing.assert_allclose(ret, [3, 2, 100])


def test_masked_gradient():
    torch.set_num_threads(1)
    env = RoutingEnv(builtin("random", 5), synthetic("line", 5), shape=Shape(8, 16, 16))
    model = JointPolicy(dim=16, layers=1)
    logits, value = model(**tensor_obs(env.observe()))
    assert torch.isfinite(logits).all()
    assert (logits[0, ~torch.as_tensor(env.action_masks())] < -1000).all()
    (-torch.log_softmax(logits, -1)[0, 0] + value.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_sabre_demonstrations():
    env = RoutingEnv(builtin("random", 5, 22, 4), synthetic("line", 5), shape=Shape(8, 16, 16))
    samples = expert_trace(env, 4)
    assert samples and env.completed
    assert all(obs["action_mask"][action] == 1 for obs, action in samples)


def test_training_smoke(tmp_path):
    config = dict(
        shape=dict(nodes=8, edges=16, tokens=16),
        model=dict(dim=16, layers=1),
        curriculum=[dict(qubits=3, steps=16)],
        topologies=["line"],
        families=["random"],
        circuit_depth=2,
        imitation_episodes=2,
        imitation_epochs=1,
        torch_threads=1,
        rollout_steps=8,
        ppo_epochs=1,
        batch_size=4,
    )
    checkpoint = imitation(config, tmp_path / "bc")
    assert ppo(config, tmp_path / "rl", checkpoint).exists()
