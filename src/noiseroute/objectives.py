"""Imitation anchoring and a terminal, action-independent SABRE reference."""

import copy
import hashlib
import math

import gymnasium as gym
import numpy as np
import torch

from .benchmark import compile_one
from .model import JointPolicy, load_model


def learning_rate(config, method):
    # Old shared learning_rate remains a fallback for existing experiment resumes.
    default = {"imitation": 5e-5, "ppo": 1e-5}[method]
    value = float(config.get(f"{method}_learning_rate", config.get("learning_rate", default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{method}_learning_rate must be finite and positive")
    return value


def reference_policy(model, shape, checkpoint=None, saved=None, device="cpu"):
    """Restore the original teacher on resume; never replace it with the PPO actor."""
    if saved:
        reference = JointPolicy(**saved["config"]).to(device)
        reference.load_state_dict(saved["state_dict"])
        identity = saved["identity"]
    else:
        if not checkpoint:
            raise ValueError("KL anchoring requires an imitation checkpoint")
        reference, ref_shape, metadata = load_model(checkpoint, device)
        if metadata.get("method") != "SABRE imitation":
            raise ValueError("Reference must be SABRE imitation, not a PPO checkpoint")
        if ref_shape != shape:
            raise ValueError("Reference shape differs from training configuration")
        with open(checkpoint, "rb") as stream:
            identity = hashlib.sha256(stream.read()).hexdigest()
    if reference.config != model.config:
        raise ValueError("Reference architecture differs from PPO actor")
    reference.eval().requires_grad_(False)
    payload = dict(
        config=copy.deepcopy(reference.config),
        state_dict={k: v.detach().cpu().clone() for k, v in reference.state_dict().items()},
        identity=identity,
    )
    return reference, payload


def reference_kl(logits, reference_logits, action_mask):
    """D_KL(pi_PPO || pi_imitation) on the identical set of legal actions."""
    mask = action_mask.bool()
    if not mask.any(dim=-1).all():
        raise ValueError("KL requires at least one legal action per observation")
    # Finite masking avoids the undefined gradient of 0 * (-inf - -inf).
    # Remove illegal entries before multiplication, not only after reduction.
    current = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    frozen = torch.log_softmax(reference_logits.detach().masked_fill(~mask, -1e9), dim=-1)
    log_ratio = (current - frozen).masked_fill(~mask, 0)
    probabilities = current.exp().masked_fill(~mask, 0)
    return (probabilities * log_ratio).sum(dim=-1).mean()


def distinct_seeds(seeds, minimum=20):
    if not isinstance(seeds, (list, tuple)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)) or len(seeds) < minimum:
        raise ValueError(f"At least {minimum} distinct seeds are required; duplicates are forbidden")
    return list(seeds)


class SabreRelativeReward(gym.Wrapper):
    """Sum of successful episode rewards = logESP(candidate) - median(logESP(SABRE)).

    Only the final reward changes; intermediate SWAP shaping telescopes exactly.
    A failed route keeps its original failure penalty, with no baseline bonus.
    """

    def __init__(self, env, seeds):
        super().__init__(env)
        self.seeds = distinct_seeds(seeds)
        if not env.exact_terminal or env.terminal_optimization_level != 3:
            raise ValueError("Comparative reward requires exact level-3 terminal ESP")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and info["completed"]:
            baseline = []
            for seed in self.seeds:
                _, metrics = compile_one(self.env.circuit, self.env.device, "sabre", seed)
                baseline.append(metrics["log_esp"])
            if not np.isfinite(baseline).all():
                raise FloatingPointError("Non-finite SABRE reward reference")
            reference = float(np.median(baseline))
            reward -= reference
            info.update(
                sabre_log_esp=reference,
                delta_log_esp=info["terminal_log_esp"] - reference,
                sabre_seeds=len(self.seeds),
            )
        return obs, reward, terminated, truncated, info
