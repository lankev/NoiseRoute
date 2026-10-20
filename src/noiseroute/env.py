"""Joint masked placement/routing environment; Rust owns routing transitions."""

import copy
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from qiskit import transpile

from .circuit import materialize, normalize, qubit_lists
from .core import core_class
from .metrics import evaluate


@dataclass
class Shape:
    nodes: int = 32
    edges: int = 96
    tokens: int = 64


class RoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        circuit,
        device,
        shape=None,
        joint=True,
        initial=None,
        backend="auto",
        max_steps=None,
        exact_terminal=True,
        terminal_optimization_level=0,
    ):
        self.circuit = normalize(circuit)
        self.device = device
        self.shape = shape or Shape(max(32, device.n), max(96, len(device.edges)), 64)
        self.gates = qubit_lists(self.circuit)
        self.dependency_gates = [
            list(range(device.n)) if i.operation.name == "measure" else g
            for i, g in zip(self.circuit.data, self.gates)
        ]
        self.joint, self.initial, self.backend = joint, initial, backend
        self.exact_terminal = exact_terminal
        self.terminal_optimization_level = terminal_optimization_level
        self.max_steps = max_steps or max(100, 20 * device.n + 10 * len(self.gates))
        n, e, t = self.shape.nodes, self.shape.edges, self.shape.tokens
        if min(n, e, t) < 1 or device.n > n or len(device.edges) > e or self.circuit.num_qubits > device.n:
            raise ValueError("Circuit/device exceeds positive shape bucket")
        self.action_space = gym.spaces.Discrete(n + e)
        self.observation_space = gym.spaces.Dict(
            {
                "nodes": gym.spaces.Box(-1, 10, (n, 8), np.float32),
                "adj": gym.spaces.Box(0, 1, (n, n), np.float32),
                "tokens": gym.spaces.Box(-1, 10, (t, 8), np.float32),
                "edges": gym.spaces.Box(0, n - 1, (e, 2), np.int64),
                "node_mask": gym.spaces.Box(0, 1, (n,), np.float32),
                "token_mask": gym.spaces.Box(0, 1, (t,), np.float32),
                "action_mask": gym.spaces.Box(0, 1, (n + e,), np.float32),
                "phase": gym.spaces.Box(0, 1, (1,), np.float32),
                "clusters": gym.spaces.Box(0, 1, (n, (n + 15) // 16), np.float32),
            }
        )
        self.order = sorted(
            range(self.circuit.num_qubits),
            key=lambda q: (-sum(q in g and len(g) == 2 for g in self.gates), q),
        )
        # Static features are cached; only circuit-dependent features change on step.
        self._adj = np.zeros((n, n), np.float32)
        self._edges = np.zeros((e, 2), np.int64)
        for idx, (a, b) in enumerate(device.edges):
            self._edges[idx] = a, b
            self._adj[a, b] = self._adj[b, a] = 1 - device.edge_error(a, b)
        np.fill_diagonal(self._adj, 1)
        self._clusters = np.zeros((n, (n + 15) // 16), np.float32)
        visited, bfs, queue = set(), [], [0]
        while queue:
            q = queue.pop(0)
            if q in visited:
                continue
            visited.add(q)
            bfs.append(q)
            queue += [int(k) for k in np.flatnonzero(self._adj[q, : device.n]) if k not in visited]
        for j, q in enumerate(bfs):
            self._clusters[q, j // 16] = 1
        self.reset()

    def _start_core(self):
        n = self.device.n
        missing = iter(sorted(set(range(n)) - set(self.placement.values())))
        mapping = [self.placement.get(q, -1) for q in range(n)]
        mapping = [next(missing) if p == -1 else p for p in mapping]
        self.initial_mapping = mapping.copy()
        self.core = core_class(self.backend)(n, self.device.edges, self.dependency_gates, mapping)
        self._drain()

    def _drain(self):
        mapping = self.core.mapping()
        for i in self.core.drain():
            self.events.append(("gate", i, [mapping[q] for q in self.gates[i]]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.placement, self.events = {}, []
        self.core = None
        self.steps, self.swap_proxy = 0, 0.0
        self.completed, self.failed = False, False
        self.terminal_log_esp = None
        if not self.joint or not self.order:
            initial = self.initial if self.initial is not None else list(range(self.circuit.num_qubits))
            if (
                len(initial) != self.circuit.num_qubits
                or len(set(initial)) != len(initial)
                or any(p < 0 or p >= self.device.n for p in initial)
            ):
                raise ValueError("Invalid initial placement")
            self.placement = dict(enumerate(initial))
            self._start_core()
        return self.observe(), {}

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.core is None:
            for p in set(range(self.device.n)) - set(self.placement.values()):
                mask[p] = True
        elif not self.core.remaining():
            mask[0] = True
        else:
            m = self.core.mask()
            mask[self.shape.nodes : self.shape.nodes + len(m)] = m
        return mask

    def _finish(self):
        self.completed = True
        if self.exact_terminal:
            native = transpile(
                self.output(),
                target=self.device.target(),
                initial_layout=list(range(self.device.n)),
                routing_method="none",
                optimization_level=self.terminal_optimization_level,
                seed_transpiler=0,
            )
            exact = evaluate(native, self.device)["log_esp"]
        else:
            exact = self.swap_proxy
        self.terminal_log_esp = exact
        return exact - self.swap_proxy

    def step(self, action):
        if self.completed or self.failed:
            raise RuntimeError("reset required after termination")
        action = int(action)
        if action < 0 or action >= self.action_space.n or not self.action_masks()[action]:
            raise ValueError("Masked action rejected")
        self.steps += 1
        reward = 0.0
        if self.core is None:
            self.placement[self.order[len(self.placement)]] = action
            if len(self.placement) == len(self.order):
                self._start_core()
        elif self.core.remaining():
            edge = action - self.shape.nodes
            a, b = self.device.edges[edge]
            reward = 3 * math.log1p(-self.device.edge_error(a, b))
            self.swap_proxy += reward
            self.events.append(("swap", edge, [a, b]))
            self.core.swap(edge)
            self._drain()
        if self.core is not None and not self.core.remaining():
            reward += self._finish()
        elif self.steps >= self.max_steps:
            self.failed = True
            reward -= 10000.0
        info = {
            "completed": self.completed,
            "failed": self.failed,
            "swaps": sum(k == "swap" for k, _, _ in self.events),
            "terminal_log_esp": self.terminal_log_esp,
        }
        return self.observe(), reward, self.completed or self.failed, False, info

    def output(self):
        if self.core is None or self.core.remaining():
            raise ValueError("Cannot emit incomplete circuit")
        return materialize(self.circuit, self.device, self.events)

    def clone(self):
        obj = copy.copy(self)
        obj.placement, obj.events = self.placement.copy(), self.events.copy()
        obj.core = self.core.clone() if self.core is not None else None
        return obj

    def observe(self):
        n, t = self.shape.nodes, self.shape.tokens
        d = self.device
        nodes, tokens = np.zeros((n, 8), np.float32), np.zeros((t, 8), np.float32)
        nm, tm = np.zeros(n, np.float32), np.zeros(t, np.float32)
        nm[: d.n] = 1
        mapping = self.core.mapping() if self.core else [self.placement.get(q, -1) for q in range(d.n)]
        front = set(self.core.front()) if self.core else set(range(min(t, len(self.gates))))
        remaining = self.core.remaining() if self.core else list(range(len(self.gates)))
        window = sorted(front) + [i for i in remaining if i not in front]
        for p in range(d.n):
            ro = d.cal.get(("measure", (p,)), {"error": 0})["error"]
            nodes[p, :5] = [
                self._adj[p, : d.n].sum() / d.n,
                ro,
                min(d.t1[p] / 0.001, 1),
                min(d.t2[p] / 0.001, 1),
                float(p in mapping),
            ]
            nodes[p, 5] = sum(mapping[q] == p for i in front for q in self.gates[i]) / max(1, len(front))
            nodes[p, 6] = self.steps / self.max_steps
            nodes[p, 7] = len(remaining) / max(1, len(self.gates))
        for j, i in enumerate(window[:t]):
            g = self.gates[i]
            a = mapping[g[0]] if g else -1
            b = mapping[g[-1]] if g else -1
            dist = d.dist[a, b] / d.n if min(a, b) >= 0 else 1
            tokens[j] = [
                a / n,
                b / n,
                len(g) / 2,
                float(i in front),
                i / max(1, len(self.gates)),
                dist,
                float(self.circuit.data[i].operation.name == "measure"),
                float(self.core is not None or self.order[len(self.placement)] in g),
            ]
            tm[j] = 1
        tm[0] = 1
        return dict(
            nodes=nodes,
            adj=self._adj.copy(),
            tokens=tokens,
            edges=self._edges.copy(),
            node_mask=nm,
            token_mask=tm,
            action_mask=self.action_masks().astype(np.float32),
            phase=np.array([self.core is not None], np.float32),
            clusters=self._clusters.copy(),
        )
