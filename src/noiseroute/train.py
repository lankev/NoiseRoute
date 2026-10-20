"""SABRE imitation, clipped PPO/GAE, MaskablePPO control and distillation."""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from torch import nn

from .benchmark import provenance
from .datasets import builtin, load_manifest
from .device import synthetic
from .env import RoutingEnv, Shape
from .model import JointPolicy, load_model, save_model, tensor_obs
from .objectives import SabreRelativeReward, distinct_seeds, learning_rate, reference_kl, reference_policy


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config, stage, episode, shape, *, topology=None, family=None):
    seed = int(config.get("seed", 0)) + episode
    topologies = config.get("topologies", ["line", "ring", "grid"])
    kind = topology or topologies[episode % len(topologies)]
    families = config.get("families", ["random", "ansatz"])
    if config.get("train_manifest"):
        records = load_manifest(config["train_manifest"], split="train")
        _, qc = records[episode % len(records)]
        n = max(stage["qubits"], qc.num_qubits)
    else:
        n = stage["qubits"]
        qc = builtin(family or families[episode % len(families)], n, seed, config.get("circuit_depth", 3))
    device = synthetic(kind, n, seed)
    if config.get("measure_all", True) and not qc.count_ops().get("measure", 0):
        qc = qc.copy()
        qc.measure_all()
    return RoutingEnv(
        qc,
        device,
        shape=shape,
        exact_terminal=True,
        terminal_optimization_level=3 if config.get("reward", {}).get("mode") == "sabre_delta_log_esp" else 0,
    )


def expert_trace(env, seed=0):
    pm = PassManager(
        SabreLayout(env.device.coupling, seed=seed, skip_routing=True, layout_trials=1, swap_trials=1)
    )
    pm.run(env.circuit)
    layout = pm.property_set["layout"]
    mapping = [layout[q] for q in env.circuit.qubits]
    embedded = QuantumCircuit(env.device.n, env.circuit.num_clbits)
    for inst in env.circuit.data:
        embedded.append(
            inst.operation,
            [mapping[env.circuit.find_bit(q).index] for q in inst.qubits],
            [env.circuit.find_bit(c).index for c in inst.clbits],
        )
    teacher = PassManager(SabreSwap(env.device.coupling, heuristic="decay", seed=seed, trials=1)).run(
        embedded
    )
    obs, _ = env.reset(seed=seed)
    samples = []
    while env.core is None:
        action = mapping[env.order[len(env.placement)]]
        samples.append((obs, action))
        obs, _, done, _, _ = env.step(action)
        if done:
            return samples
    for inst in teacher.data:
        if inst.operation.name != "swap" or env.completed:
            continue
        edge = tuple(sorted(teacher.find_bit(q).index for q in inst.qubits))
        action = env.shape.nodes + env.device.edges.index(edge)
        if not env.action_masks()[action]:
            return []
        samples.append((obs, action))
        obs, _, _, _, _ = env.step(action)
    if not env.completed and not env.core.remaining():
        env.step(0)
    return samples if env.completed else []


def batch_observations(samples, device):
    return tensor_obs({k: np.stack([s[k] for s in samples]) for k in samples[0]}, device, batch=False)


def demonstration_case(config, episode):
    """Visit the Cartesian product without correlating size/topology/family."""
    stages = config.get("curriculum", [{"qubits": 5, "steps": 1024}])
    topologies = config.get("topologies", ["line", "ring", "grid"])
    families = config.get("families", ["random", "ansatz"])
    block = len(topologies) * len(families)
    return (
        stages[(episode // block) % len(stages)],
        topologies[episode % len(topologies)],
        families[(episode // len(topologies)) % len(families)],
    )


def training_rewards(rewards, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reward_scale must be finite and positive")
    return np.asarray(rewards, dtype=np.float32) * scale


def value_loss(values, returns, kind):
    if kind == "mse":
        return nn.functional.mse_loss(values, returns)
    if kind == "huber":
        return nn.functional.smooth_l1_loss(values, returns)
    raise ValueError("value_loss must be 'mse' or 'huber'")


def quality_status(outcomes, gate):
    if not gate:
        return dict(status="disabled")
    window = int(gate.get("window", 100))
    minimum = int(gate.get("min_episodes", 30))
    maximum_failure = float(gate.get("max_failure_rate", 0.25))
    minimum_success = float(gate.get("min_success_rate", 0.9))
    if not 0 < minimum <= window or not 0 <= maximum_failure <= 1 or not 0 <= minimum_success <= 1:
        raise ValueError("Invalid quality_gate configuration")
    recent = outcomes[-window:]
    if len(recent) < minimum:
        return dict(status="insufficient", episodes=len(recent))
    success = sum(recent) / len(recent)
    return dict(
        status="pass" if success >= minimum_success else "below_target",
        episodes=len(recent),
        success_rate=success,
        collapse=(1 - success) > maximum_failure,
    )


class TrainingQualityError(RuntimeError):
    """Training stopped after saving because an explicit quality gate failed."""


class Progress:
    """Rate-limited console output; PPO also exposes a small status.json."""

    def __init__(self, name, interval=10, root=None):
        self.name, self.interval, self.root = name, max(0, float(interval)), root
        self.started, self.last = time.monotonic(), float("-inf")
        self.fields = {}

    def report(self, phase, force=False, **fields):
        self.fields.update(fields)
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        record = dict(phase=phase, elapsed_s=round(now - self.started, 1), **self.fields)
        if self.root:
            temporary = self.root / "status.json.tmp"
            temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temporary.replace(self.root / "status.json")
        details = " ".join(f"{k}={v}" for k, v in record.items())
        print(f"[{self.name}] {details}", flush=True)


def imitation(config, output):
    seed_all(config.get("seed", 0))
    torch.set_num_threads(config.get("torch_threads", 4))
    device = config.get("torch_device", "cpu")
    shape = Shape(**config.get("shape", {}))
    model = JointPolicy(**config.get("model", {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(config, "imitation"))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    progress = Progress("IMITATION", config.get("progress_interval_seconds", 10))
    progress.report("demonstrations", force=True, device=device)
    samples, rejected = [], 0
    coverage = defaultdict(lambda: dict(episodes=0, accepted=0, samples=0))
    for episode in range(config.get("imitation_episodes", 64)):
        stage, topology, family = demonstration_case(config, episode)
        env = make_env(config, stage, episode, shape, topology=topology, family=family)
        trace = expert_trace(env, config.get("seed", 0) + episode)
        rejected += not bool(trace)
        samples.extend(trace)
        entry = coverage[f"{stage['qubits']}/{topology}/{family}"]
        entry["episodes"] += 1
        entry["accepted"] += bool(trace)
        entry["samples"] += len(trace)
        progress.report("demonstrations", episodes=episode + 1, samples=len(samples), rejected=rejected)
    (root / "coverage.json").write_text(json.dumps(dict(coverage), indent=2), encoding="utf-8")
    if config.get("require_demo_coverage", False):
        expected = {
            f"{s['qubits']}/{t}/{f}"
            for s in config["curriculum"]
            for t in config.get("topologies", ["line", "ring", "grid"])
            for f in config.get("families", ["random", "ansatz"])
        }
        missing = sorted(k for k in expected if not coverage.get(k, {}).get("accepted", 0))
        if missing:
            raise ValueError(f"Missing complete demonstrations: {missing}; inspect {root / 'coverage.json'}")
    if not samples:
        raise ValueError("No complete replayable SABRE demonstrations")
    batch_size = config.get("batch_size", 32)
    losses = []
    for epoch in range(config.get("imitation_epochs", 5)):
        progress.report("optimize", force=True, epoch=epoch + 1, samples=len(samples))
        order = np.random.permutation(len(samples))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            data = [samples[i] for i in order[start : start + batch_size]]
            logits, _ = model(**batch_observations([s[0] for s in data], device))
            actions = torch.tensor([s[1] for s in data], device=device)
            loss = nn.functional.cross_entropy(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            progress.report("optimize", batch=start // batch_size + 1, loss=round(epoch_losses[-1], 5))
        losses.append(float(np.mean(epoch_losses)))
        progress.report("epoch_done", force=True, mean_loss=round(losses[-1], 5))
    metadata = dict(
        method="SABRE imitation",
        learning_rate=learning_rate(config, "imitation"),
        samples=len(samples),
        rejected_episodes=rejected,
        coverage=dict(coverage),
        losses=losses,
        config=config,
        environment=provenance(),
        research_validated=False,
    )
    save_model(root / "imitation.pt", model, shape, metadata)
    (root / "imitation.json").write_text(json.dumps(metadata, indent=2))
    progress.report("saved", force=True, checkpoint=str(root / "imitation.pt"))
    return root / "imitation.pt"


def gae(rewards, values, dones, bootstrap, gamma=1.0, lam=0.95):
    advantages = np.zeros(len(rewards), np.float32)
    last = 0.0
    for i in reversed(range(len(rewards))):
        next_value = bootstrap if i == len(rewards) - 1 else values[i + 1]
        live = 1 - float(dones[i])
        delta = rewards[i] + gamma * next_value * live - values[i]
        last = delta + gamma * lam * live * last
        advantages[i] = last
    return advantages, advantages + np.asarray(values)


def ppo(config, output, checkpoint=None, resume=False):
    if resume and not checkpoint:
        raise ValueError("--resume requires a PPO --checkpoint")
    seed_all(config.get("seed", 0))
    torch.set_num_threads(config.get("torch_threads", 4))
    device = config.get("torch_device", "cpu")
    kl_coef = float(config.get("reference_kl_coef", 0))
    if not np.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("reference_kl_coef must be finite and non-negative")
    reward_config = config.get("reward", {})
    reward_mode = reward_config.get("mode", "absolute_log_esp")
    if reward_mode not in ("absolute_log_esp", "sabre_delta_log_esp"):
        raise ValueError("Unknown reward mode")
    comparative = reward_mode == "sabre_delta_log_esp"
    if comparative:
        reward_seeds = distinct_seeds(reward_config.get("seeds", list(range(20))))
        if float(config.get("gamma", 1)) != 1:
            raise ValueError("Comparative episodic logESP requires gamma=1")
    validation = config.get("validation")
    if (comparative or kl_coef) and not validation:
        raise ValueError("Anchored/comparative PPO requires a >=20-seed validation configuration")
    if validation:
        from .acceptance import validate_checkpoint, validation_config

        validation = validation_config(validation)
    reward_scale = float(config.get("reward_scale", 1.0))
    training_rewards([], reward_scale)
    value_kind = config.get("value_loss", "mse")
    if value_kind not in ("mse", "huber"):
        raise ValueError("value_loss must be 'mse' or 'huber'")
    gate = config.get("quality_gate", {})
    quality_status([], gate)
    target_kl = config.get("target_kl")
    if target_kl is not None and (not np.isfinite(target_kl) or target_kl <= 0):
        raise ValueError("target_kl must be positive")
    shape = Shape(**config.get("shape", {}))
    metadata, training_state = {}, {}
    if checkpoint:
        model, saved_shape, metadata = load_model(checkpoint, device)
        if saved_shape != shape:
            raise ValueError("Checkpoint shape differs from training configuration")
    else:
        model = JointPolicy(**config.get("model", {})).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(config, "ppo"))
    total_steps, episode = 0, 0
    if resume:
        if metadata.get("method") != "PPO":
            raise ValueError("--resume requires a PPO checkpoint; use --checkpoint alone for imitation")
        ignored = {"torch_device", "torch_threads", "progress_interval_seconds"}
        previous = {k: v for k, v in metadata.get("config", {}).items() if k not in ignored}
        current = {k: v for k, v in config.items() if k not in ignored}
        if previous != current:
            raise ValueError("Resume requires the same training configuration and curriculum")
        total_steps = int(metadata["steps"])
        training_state = torch.load(checkpoint, map_location="cpu", weights_only=True).get(
            "training_state", {}
        )
        if training_state.get("optimizer"):
            optimizer.load_state_dict(training_state["optimizer"])
        else:
            print("[PPO] Old checkpoint: weights and step count restored; Adam starts fresh.", flush=True)
        # A new episode avoids pretending we restored the in-flight DAG/RNG.
        episode = int(training_state.get("episode", total_steps)) + 1
        print("[PPO] Resume starts a fresh episode; it is not a bit-for-bit continuation.", flush=True)
    reference, reference_state = None, None
    if kl_coef:
        if resume and not training_state.get("imitation_reference"):
            raise ValueError("Resume lacks the frozen imitation reference; cannot safely re-anchor PPO")
        reference, reference_state = reference_policy(
            model,
            shape,
            config.get("reference_checkpoint", checkpoint),
            training_state.get("imitation_reference") if resume else None,
            device,
        )
    target = sum(stage["steps"] for stage in config["curriculum"])
    if not 0 <= total_steps <= target:
        raise ValueError("Saved step count is outside the configured curriculum")
    saved_quality = metadata.get("training_quality", {})
    boundaries = set(np.cumsum([s["steps"] for s in config["curriculum"]]).tolist())
    if (
        resume
        and gate
        and (
            saved_quality.get("collapse")
            or (total_steps in boundaries and saved_quality.get("status") != "pass")
        )
    ):
        raise TrainingQualityError(
            "Checkpoint did not pass its quality gate; inspect its log before a new experiment"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("last.pt", "train.json", "status.json")):
        raise FileExistsError("Choose a new output directory to preserve the previous training run")
    progress = Progress("PPO", config.get("progress_interval_seconds", 10), root)
    start_steps = saved_steps = total_steps
    recent_outcomes = training_state.get("recent_outcomes", []) if resume else []
    history_stage = training_state.get("history_stage") if resume else None
    log, environment = [], provenance()
    validation_log = []
    next_validation = (
        ((total_steps // validation["every_steps"]) + 1) * validation["every_steps"] if validation else None
    )
    last_validated = None

    def environment_for(stage, episode):
        env = make_env(config, stage, episode, shape)
        return SabreRelativeReward(env, reward_seeds) if comparative else env

    def validate():
        nonlocal last_validated, next_validation
        progress.report("validating", force=True, step=total_steps, seeds=len(validation["seeds"]))
        directory = root / f"validation-{total_steps:09d}"
        report = validate_checkpoint(
            root / "last.pt",
            validation,
            directory,
            progress=lambda **fields: progress.report("validating", **fields),
        )
        entry = dict(step=total_steps, accepted=report["accepted"], report=str(directory / "report.json"))
        validation_log.append(entry)
        (root / "validation.json").write_text(json.dumps(validation_log, indent=2), encoding="utf-8")
        if report["accepted"]:
            temporary = root / "accepted.pt.tmp"
            temporary.write_bytes((directory / "accepted.pt").read_bytes())
            temporary.replace(root / "accepted.pt")
            (root / "accepted.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        last_validated = total_steps
        next_validation = ((total_steps // validation["every_steps"]) + 1) * validation["every_steps"]
        progress.report("accepted" if report["accepted"] else "rejected", force=True, **entry)

    progress.report(
        "start",
        force=True,
        device=device,
        step=total_steps,
        target=target,
        saved_step=saved_steps,
        source=str(checkpoint) if checkpoint else None,
    )

    def save(path):
        save_model(
            path,
            model,
            shape,
            dict(
                method="PPO",
                learning_rate=learning_rate(config, "ppo"),
                reference_kl_coef=kl_coef,
                imitation_reference_sha256=reference_state["identity"] if reference_state else None,
                checkpoint_role="recovery_only; acceptance requires matching validation report",
                steps=total_steps,
                config=config,
                environment=environment,
                research_validated=False,
                resumed_from=str(checkpoint) if resume else None,
                training_quality=quality_status(recent_outcomes, gate),
            ),
            training_state=dict(
                optimizer=optimizer.state_dict(),
                episode=episode,
                recent_outcomes=recent_outcomes,
                history_stage=history_stage,
                imitation_reference=reference_state,
            ),
        )

    stage_start = 0
    try:
        if total_steps == target:
            save(root / "last.pt")
        for stage_index, stage in enumerate(config["curriculum"]):
            stage_end = stage_start + stage["steps"]
            if total_steps >= stage_end:
                stage_start = stage_end
                continue
            stage_steps = total_steps - stage_start
            if history_stage != stage_index:
                recent_outcomes = []
                history_stage = stage_index
            progress.report("new_stage", force=True, stage=stage_index + 1, qubits=stage["qubits"])
            env = environment_for(stage, episode)
            obs, _ = env.reset()
            while stage_steps < stage["steps"]:
                collected_at = time.monotonic()
                observations, actions, old_logps, rewards, values, dones = [], [], [], [], [], []
                completed, failed = 0, 0
                deltas, references = [], []
                progress.report("collect", force=True, step=total_steps, saved_step=saved_steps)
                for _ in range(min(config.get("rollout_steps", 256), stage["steps"] - stage_steps)):
                    with torch.no_grad():
                        logits, value = model(**tensor_obs(obs, device))
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                    observations.append(obs)
                    actions.append(int(action.item()))
                    old_logps.append(float(dist.log_prob(action).item()))
                    values.append(float(value.item()))
                    obs, reward, done, _, info = env.step(actions[-1])
                    rewards.append(reward)
                    dones.append(done)
                    stage_steps += 1
                    total_steps += 1
                    if done:
                        if "delta_log_esp" in info:
                            deltas.append(info["delta_log_esp"])
                            references.append(info["sabre_log_esp"])
                        completed += info["completed"]
                        failed += info["failed"]
                        recent_outcomes.append(int(bool(info["completed"])))
                        recent_outcomes = recent_outcomes[-int(gate.get("window", 100)) :]
                        episode += 1
                        env = environment_for(stage, episode)
                        obs, _ = env.reset()
                    progress.report("collect", step=total_steps)
                with torch.no_grad():
                    bootstrap = float(model(**tensor_obs(obs, device))[1].item())
                adv, returns = gae(
                    training_rewards(rewards, reward_scale),
                    values,
                    dones,
                    bootstrap,
                    config.get("gamma", 1.0),
                    config.get("gae_lambda", 0.95),
                )
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                collect_seconds = time.monotonic() - collected_at
                update_at = time.monotonic()
                progress.report("optimize", force=True, step=total_steps)
                # Transfer this rollout once, rather than reconstructing/copying
                # all nine observation arrays for every minibatch in every epoch.
                tensors = batch_observations(observations, device)
                action_tensor = torch.as_tensor(actions, device=device)
                old_tensor = torch.as_tensor(old_logps, dtype=torch.float32, device=device)
                advantage_tensor = torch.as_tensor(adv, device=device)
                return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
                reference_logits = None
                if reference is not None:
                    with torch.no_grad():
                        reference_logits = torch.cat(
                            [
                                reference(
                                    **{k: t[i : i + config.get("batch_size", 32)] for k, t in tensors.items()}
                                )[0]
                                for i in range(0, len(actions), config.get("batch_size", 32))
                            ]
                        )
                losses = []
                actor_losses, critic_losses, entropies, kls, clip_fractions = [], [], [], [], []
                stopped_kl = False
                for epoch in range(config.get("ppo_epochs", 4)):
                    order = np.random.permutation(len(actions))
                    for start in range(0, len(actions), config.get("batch_size", 32)):
                        idx = torch.as_tensor(
                            order[start : start + config.get("batch_size", 32)], device=device
                        )
                        logits, v = model(**{k: t[idx] for k, t in tensors.items()})
                        dist = torch.distributions.Categorical(logits=logits)
                        log_ratio = dist.log_prob(action_tensor[idx]) - old_tensor[idx]
                        ratio = torch.exp(log_ratio)
                        approx_kl = float((ratio - 1 - log_ratio).mean().detach())
                        if not np.isfinite(approx_kl):
                            raise FloatingPointError("Non-finite PPO KL; last complete checkpoint preserved")
                        kls.append(approx_kl)
                        if target_kl is not None and approx_kl > 1.5 * target_kl:
                            stopped_kl = True
                            break
                        advantage = advantage_tensor[idx]
                        clip = config.get("clip_range", 0.2)
                        actor = -torch.minimum(
                            ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
                        ).mean()
                        critic = value_loss(v, return_tensor[idx], value_kind)
                        entropy = dist.entropy().mean()
                        anchored_kl = (
                            reference_kl(logits, reference_logits[idx], tensors["action_mask"][idx])
                            if reference is not None
                            else logits.new_zeros(())
                        )
                        loss = (
                            actor
                            + config.get("value_coef", 0.5) * critic
                            - config.get("entropy_coef", 0.01) * entropy
                            + kl_coef * anchored_kl
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "Non-finite PPO loss; last complete checkpoint preserved"
                            )
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        actor_losses.append(float(actor.detach()))
                        critic_losses.append(float(critic.detach()))
                        entropies.append(float(entropy.detach()))
                        clip_fractions.append(float(((ratio - 1).abs() > clip).float().mean().detach()))
                        progress.report(
                            "optimize", epoch=epoch + 1, batch=start // config.get("batch_size", 32) + 1
                        )
                    if stopped_kl:
                        break
                post_kl = None
                if reference is not None:
                    with torch.no_grad():
                        post_logits = torch.cat(
                            [
                                model(
                                    **{k: t[i : i + config.get("batch_size", 32)] for k, t in tensors.items()}
                                )[0]
                                for i in range(0, len(actions), config.get("batch_size", 32))
                            ]
                        )
                        post_kl = float(reference_kl(post_logits, reference_logits, tensors["action_mask"]))
                        if not np.isfinite(post_kl):
                            raise FloatingPointError("Non-finite imitation KL; checkpoint not saved")
                del tensors, action_tensor, old_tensor, advantage_tensor, return_tensor, reference_logits
                update_seconds = time.monotonic() - update_at
                log.append(
                    dict(
                        step=total_steps,
                        stage=stage_index,
                        completed=completed,
                        failed=failed,
                        mean_reward=float(np.mean(rewards)),
                        mean_training_reward=float(np.mean(training_rewards(rewards, reward_scale))),
                        loss=float(np.mean(losses)) if losses else None,
                        actor_loss=float(np.mean(actor_losses)) if actor_losses else None,
                        value_loss=float(np.mean(critic_losses)) if critic_losses else None,
                        entropy=float(np.mean(entropies)) if entropies else None,
                        approx_kl=float(np.mean(kls)),
                        imitation_kl=post_kl,
                        reference_kl_coef=kl_coef,
                        mean_delta_log_esp=float(np.mean(deltas)) if deltas else None,
                        mean_sabre_log_esp=float(np.mean(references)) if references else None,
                        clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else None,
                        kl_early_stop=stopped_kl,
                        optimizer_steps=len(losses),
                        quality=quality_status(recent_outcomes, gate),
                        collect_seconds=collect_seconds,
                        update_seconds=update_seconds,
                    )
                )
                # Save only a fully updated policy. Ctrl+C during the next update
                # cannot replace it with a partially updated model.
                save(root / "last.pt")
                saved_steps = total_steps
                (root / "train.json").write_text(
                    json.dumps(
                        dict(
                            config=config,
                            resumed_from=str(checkpoint) if resume else None,
                            starting_step=start_steps,
                            updates=log,
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                elapsed = max(time.monotonic() - progress.started, 1e-9)
                rate = (total_steps - start_steps) / elapsed
                progress.report(
                    "saved",
                    force=True,
                    step=total_steps,
                    saved_step=saved_steps,
                    percent=round(100 * total_steps / max(target, 1), 2),
                    steps_per_s=round(rate, 2),
                    completed=completed,
                    failed=failed,
                )
                quality = quality_status(recent_outcomes, gate)
                at_boundary = stage_steps == stage["steps"]
                if gate and (quality.get("collapse") or (at_boundary and quality["status"] != "pass")):
                    progress.report("quality_stop", force=True, quality=quality)
                    raise TrainingQualityError(
                        f"Quality gate stopped training: {quality}. Saved: {root / 'last.pt'}"
                    )
                if validation and (at_boundary or total_steps >= next_validation):
                    validate()
            save(root / f"stage_{stage_index}.pt")
            stage_start = stage_end
        if validation and last_validated != total_steps:
            validate()
    except KeyboardInterrupt:
        progress.report("interrupted", force=True, step=total_steps, saved_step=saved_steps)
        recover = root / "last.pt" if (root / "last.pt").exists() else checkpoint
        print(f"[PPO] Interrupted. Last complete checkpoint: {recover or 'none yet'}", flush=True)
        raise
    progress.report("finished", force=True, step=total_steps, saved_step=total_steps)
    return root / "last.pt"


def sb3_baseline(config, output):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    shape = Shape(**config.get("shape", {}))

    class Curriculum(gym.Wrapper):
        def __init__(self):
            self.episode = 0
            super().__init__(make_env(config, config["curriculum"][0], 0, shape))

        def reset(self, **kwargs):
            self.episode += 1
            self.env = make_env(config, config["curriculum"][0], self.episode, shape)
            return self.env.reset(**kwargs)

        def action_masks(self):
            return self.env.action_masks()

    torch.set_num_threads(config.get("torch_threads", 4))
    model = MaskablePPO(
        "MultiInputPolicy",
        DummyVecEnv([Curriculum]),
        n_steps=config.get("rollout_steps", 256),
        batch_size=config.get("batch_size", 32),
        device=config.get("torch_device", "cpu"),
        seed=config.get("seed", 0),
        gamma=config.get("gamma", 1.0),
        verbose=1,
    )
    model.learn(total_timesteps=config["curriculum"][0]["steps"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def distill(config, teacher_path, output, episodes=16):
    teacher, shape, _ = load_model(teacher_path)
    seed_all(config.get("seed", 0))
    torch.set_num_threads(config.get("torch_threads", 4))
    student = JointPolicy(dim=32, layers=1, hierarchical=False)
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    samples = []
    for episode in range(episodes):
        env = make_env(
            config, config["curriculum"][episode % len(config["curriculum"])], episode + 50000, shape
        )
        obs, _ = env.reset()
        for _ in range(min(env.max_steps, 512)):
            with torch.no_grad():
                logits, _ = teacher(**tensor_obs(obs))
            samples.append((obs, logits[0].clone()))
            obs, _, done, _, _ = env.step(int(logits.argmax()))
            if done:
                break
    for _ in range(config.get("imitation_epochs", 5)):
        for start in range(0, len(samples), 32):
            batch = samples[start : start + 32]
            predicted, _ = student(**batch_observations([b[0] for b in batch], "cpu"))
            target = torch.stack([b[1] for b in batch])
            loss = (
                nn.functional.kl_div(
                    nn.functional.log_softmax(predicted / 2, -1),
                    nn.functional.softmax(target / 2, -1),
                    reduction="batchmean",
                )
                * 4
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    save_model(
        output,
        student,
        shape,
        dict(
            method="distillation", teacher=str(teacher_path), samples=len(samples), research_validated=False
        ),
    )
    return output
