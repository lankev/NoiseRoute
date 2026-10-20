import math
import time

import numpy as np

from .circuit import verify_trace
from .env import RoutingEnv


def noise_layout(circuit, device, seed=0):
    rng = np.random.default_rng(seed)
    interactions = np.zeros((circuit.num_qubits, circuit.num_qubits))
    for inst in circuit.data:
        if len(inst.qubits) == 2:
            a, b = [circuit.find_bit(q).index for q in inst.qubits]
            interactions[a, b] += 1
            interactions[b, a] += 1
    order = sorted(range(circuit.num_qubits), key=lambda q: -interactions[q].sum())
    quality = np.zeros(device.n)
    for q in range(device.n):
        errs = [device.edge_error(a, b) for a, b in device.edges if q in (a, b)]
        quality[q] = np.mean(errs) + device.cal.get(("measure", (q,)), {"error": 0})["error"]
    mapping = {}
    for q in order:
        candidates = set(range(device.n)) - set(mapping.values())
        p = min(
            candidates,
            key=lambda p: (
                quality[p] * 10
                + sum(interactions[q, other] * device.dist[p, loc] for other, loc in mapping.items())
                + rng.random() * 1e-8
            ),
        )
        mapping[q] = p
    return [mapping[q] for q in range(circuit.num_qubits)]


def heuristic_scores(env):
    scores = np.full(env.action_space.n, -1e9, np.float32)
    if env.core is None:
        q = env.order[len(env.placement)]
        if not hasattr(env, "_heuristic_placement"):
            env._heuristic_placement = noise_layout(env.circuit, env.device)
        desired = env._heuristic_placement[q]
        for p in np.flatnonzero(env.action_masks()):
            scores[p] = -env.device.dist[p, desired]
        return scores
    if not env.core.remaining():
        scores[0] = 0
        return scores
    front, remaining = env.core.front(), env.core.remaining()[: env.shape.tokens]
    for action in np.flatnonzero(env.action_masks()):
        idx = int(action - env.shape.nodes)
        clone = env.core.clone()
        clone.swap(idx)
        m = clone.mapping()
        distance = sum(
            env.device.dist[m[env.gates[i][0]], m[env.gates[i][1]]] for i in front if len(env.gates[i]) == 2
        )
        look = sum(
            env.device.dist[m[env.gates[i][0]], m[env.gates[i][1]]]
            for i in remaining
            if len(env.gates[i]) == 2
        ) / max(1, len(remaining))
        a, b = env.device.edges[idx]
        repeat = bool(env.events and env.events[-1][0] == "swap" and env.events[-1][1] == idx)
        scores[action] = -distance - 0.5 * look + 3 * math.log1p(-env.device.edge_error(a, b)) - 0.3 * repeat
    return scores


def shortest_action(env):
    i = next(i for i in env.core.front() if len(env.gates[i]) == 2)
    a, b = [env.core.mapping()[q] for q in env.gates[i]]
    for idx, (u, v) in enumerate(env.device.edges):
        neighbor = v if u == a else u if v == a else None
        if neighbor is not None and env.device.dist[neighbor, b] < env.device.dist[a, b]:
            return env.shape.nodes + idx
    raise RuntimeError("No path")


def beam_action(env, scorer, width=4, depth=2):
    if width <= 1 or env.core is None:
        return int(np.argmax(scorer(env)))
    beams = [(0.0, env.clone(), None)]
    for _ in range(depth):
        expanded = []
        for score, state, first in beams:
            if state.completed:
                expanded.append((score, state, first))
                continue
            logits = scorer(state)
            valid = np.flatnonzero(state.action_masks())
            vals = logits[valid]
            logz = float(np.max(vals) + np.log(np.exp(vals - np.max(vals)).sum()))
            for action in valid[np.argsort(vals)[-width:]]:
                child = state.clone()
                _, reward, _, _, _ = child.step(int(action))
                expanded.append(
                    (
                        score + reward + 0.05 * (float(logits[action]) - logz),
                        child,
                        int(action) if first is None else first,
                    )
                )
        beams = sorted(expanded, key=lambda x: x[0], reverse=True)[:width]
    return beams[0][2]


def route(circuit, device, policy=None, joint=True, initial=None, beam=1, shape=None):
    start = time.perf_counter()
    env = RoutingEnv(circuit, device, shape=shape, joint=joint, initial=initial, exact_terminal=False)
    scorer = heuristic_scores if policy is None else lambda e: policy(e.observe())
    stalled, previous, fallback = 0, len(env.gates), 0
    while not env.completed and not env.failed:
        if env.core is not None and env.core.remaining() and stalled >= 2 * device.n:
            action = shortest_action(env)
            fallback += 1
        else:
            action = beam_action(env, scorer, width=beam)
        env.step(action)
        if env.core is not None:
            remaining = len(env.core.remaining())
            stalled = stalled + 1 if remaining == previous else 0
            previous = remaining
    if env.failed:
        raise RuntimeError("Routing budget exceeded; no partial circuit returned")
    verify_trace(env.circuit, device, env.events, env.initial_mapping, env.core.mapping())
    return env.output(), dict(
        initial=env.initial_mapping,
        final=env.core.mapping(),
        swaps=sum(k == "swap" for k, _, _ in env.events),
        fallback_steps=fallback,
        route_ms=(time.perf_counter() - start) * 1000,
        policy="heuristic" if policy is None else "learned",
        events=env.events,
    )
