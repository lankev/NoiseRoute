"""Python oracle for the Rust DAG/permutation engine."""

import copy
import os


class PythonCore:
    def __init__(self, n, edges, gates, mapping):
        if len(mapping) != n or sorted(mapping) != list(range(n)):
            raise ValueError("mapping must be a complete permutation")
        if any(a < 0 or b < 0 or a >= n or b >= n or a == b for a, b in edges):
            raise ValueError("Invalid edge")
        self.n, self.edges, self.gates = n, [tuple(e) for e in edges], [list(g) for g in gates]
        self.adj = {tuple(sorted(e)) for e in edges}
        self._mapping = list(mapping)
        self.inverse = [self._mapping.index(i) for i in range(n)]
        self.deps, self.children = [], [[] for _ in gates]
        last = {}
        for i, g in enumerate(gates):
            if any(q < 0 or q >= n for q in g) or len(g) != len(set(g)):
                raise ValueError("invalid gate qubits")
            deps = {last[q] for q in g if q in last}
            self.deps.append(len(deps))
            for p in deps:
                self.children[p].append(i)
            for q in g:
                last[q] = i
        self.done = [False] * len(gates)

    def mapping(self):
        return self._mapping.copy()

    def front(self):
        return [i for i, d in enumerate(self.deps) if d == 0 and not self.done[i]]

    def remaining(self):
        return [i for i, done in enumerate(self.done) if not done]

    def drain(self):
        executed = []
        while True:
            ready = [
                i
                for i in self.front()
                if len(self.gates[i]) != 2
                or tuple(sorted(self._mapping[q] for q in self.gates[i])) in self.adj
            ]
            if not ready:
                return executed
            for i in ready:
                self.done[i] = True
                executed.append(i)
                for c in self.children[i]:
                    self.deps[c] -= 1

    def mask(self):
        involved = {self._mapping[q] for i in self.front() for q in self.gates[i]}
        return [a in involved or b in involved for a, b in self.edges]

    def swap(self, action):
        if not 0 <= action < len(self.edges):
            raise ValueError("invalid edge action")
        a, b = self.edges[action]
        x, y = self.inverse[a], self.inverse[b]
        self._mapping[x], self._mapping[y] = b, a
        self.inverse[a], self.inverse[b] = y, x

    def clone(self):
        return copy.deepcopy(self)


def core_class(backend="auto"):
    backend = os.environ.get("NOISEROUTE_CORE", backend)
    if backend not in ("auto", "rust", "python"):
        raise ValueError(backend)
    if backend != "python":
        try:
            from noiseroute_core import RoutingCore

            return RoutingCore
        except ImportError:
            if backend == "rust":
                raise
    return PythonCore
