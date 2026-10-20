"""Serializable calibration snapshots; times are seconds."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler import CouplingMap, InstructionProperties, QubitProperties, Target


@dataclass
class Device:
    name: str
    n: int
    edges: list
    instructions: list[dict]
    t1: list[float]
    t2: list[float]
    source: str = "synthetic; not a hardware calibration"

    def __post_init__(self):
        self.edges = sorted({tuple(sorted(e)) for e in self.edges})
        if self.n < 2 or any(a == b or a < 0 or b >= self.n for a, b in self.edges):
            raise ValueError("Invalid device graph")
        self.dist = np.full((self.n, self.n), np.inf)
        np.fill_diagonal(self.dist, 0)
        for a, b in self.edges:
            self.dist[a, b] = self.dist[b, a] = 1
        for k in range(self.n):
            self.dist = np.minimum(self.dist, self.dist[:, k, None] + self.dist[None, k, :])
        if not np.isfinite(self.dist).all():
            raise ValueError("Connected devices only; select and reindex a connected component")
        if (
            len(self.t1) != self.n
            or len(self.t2) != self.n
            or any(not math.isfinite(t) or t <= 0 for t in self.t1 + self.t2)
        ):
            raise ValueError("One finite positive T1/T2 per physical qubit required")
        for x in self.instructions:
            if not 0 <= x["error"] < 1 or not math.isfinite(x["duration"]) or x["duration"] < 0:
                raise ValueError("Invalid calibration")
        self.cal = {(x["name"], tuple(x["qubits"])): x for x in self.instructions}
        self._edge_error, self._edge_duration = {}, {}
        for a, b in self.edges:
            vals = [v for (_, q), v in self.cal.items() if len(q) == 2 and set(q) == {a, b}]
            if not vals:
                raise ValueError(f"Missing calibration for edge {a, b}")
            self._edge_error[(a, b)] = float(np.mean([v["error"] for v in vals]))
            self._edge_duration[(a, b)] = float(np.mean([v["duration"] for v in vals]))

    @property
    def coupling(self):
        return CouplingMap([list(e) for a, b in self.edges for e in [(a, b), (b, a)]])

    def edge_error(self, a, b):
        return self._edge_error[tuple(sorted((a, b)))]

    def target(self):
        target = Target(
            num_qubits=self.n,
            qubit_properties=[QubitProperties(t1=a, t2=b) for a, b in zip(self.t1, self.t2)],
        )
        gates = get_standard_gate_name_mapping()
        gates["measure"] = Measure()
        for name in sorted({x["name"] for x in self.instructions}):
            if name not in gates:
                raise ValueError(f"Unsupported native gate {name}")
            gate = gates[name]
            if gate.is_parameterized():
                gate = gate.to_mutable()
                gate.params = [Parameter(f"p{i}") for i in range(len(gate.params))]
            props = {
                tuple(x["qubits"]): InstructionProperties(error=x["error"], duration=x["duration"])
                for x in self.instructions
                if x["name"] == name
            }
            target.add_instruction(gate, props)
        return target

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_target(cls, target, name="target"):
        instructions, excluded = [], []
        for op in target.operation_names:
            if op in ("delay", "reset"):
                continue
            for qs, p in target[op].items():
                if qs is None or len(qs) > 2:
                    continue
                if p is None or p.error is None or p.duration is None:
                    raise ValueError(f"Missing error/duration: {op} {qs}")
                if p.error == 1:
                    excluded.append(f"{op}{qs}")
                    continue
                instructions.append(dict(name=op, qubits=list(qs), error=p.error, duration=p.duration))
        qp = target.qubit_properties
        if qp is None or any(q is None or q.t1 is None or q.t2 is None for q in qp):
            raise ValueError("Missing T1/T2; provide a calibrated snapshot or explicitly synthetic device")
        return cls(
            name,
            target.num_qubits,
            [x["qubits"] for x in instructions if len(x["qubits"]) == 2],
            instructions,
            [q.t1 for q in qp],
            [q.t2 for q in qp],
            "Qiskit Target snapshot; excluded error=1 instructions: " + ", ".join(excluded),
        )


def synthetic(kind="ring", n=5, seed=0, edges=None):
    rng = np.random.default_rng(seed)
    if edges is None:
        if kind == "ring":
            edges = [(i, (i + 1) % n) for i in range(n)]
        elif kind == "line":
            edges = [(i, i + 1) for i in range(n - 1)]
        elif kind in ("grid", "neutral"):
            width = math.ceil(math.sqrt(n))
            pos = [(i % width, i // width) for i in range(n)]
            radius = 1.01 if kind == "grid" else 1.5
            edges = [(a, b) for a in range(n) for b in range(a + 1, n) if math.dist(pos[a], pos[b]) <= radius]
        elif kind == "heavy_hex":
            d = 3
            while CouplingMap.from_heavy_hex(d).size() < n:
                d += 2
            cmap = CouplingMap.from_heavy_hex(d)
            n, edges = cmap.size(), cmap.get_edges()
        else:
            raise ValueError(kind)
    edges = sorted({tuple(sorted(e)) for e in edges})
    ins = []
    for q in range(n):
        for gate in ("rz", "sx", "x", "measure"):
            error = (
                0.0
                if gate == "rz"
                else float(rng.uniform(0.01, 0.04) if gate == "measure" else rng.uniform(0.0001, 0.0008))
            )
            duration = 0.0 if gate == "rz" else (1e-6 if gate == "measure" else 35e-9)
            ins.append(dict(name=gate, qubits=[q], error=error, duration=duration))
    for a, b in edges:
        for u, v in ((a, b), (b, a)):
            ins.append(
                dict(name="cx", qubits=[u, v], error=float(rng.uniform(0.003, 0.025)), duration=300e-9)
            )
    return Device(
        f"{kind}_{n}_s{seed}",
        n,
        edges,
        ins,
        rng.uniform(90e-6, 180e-6, n).tolist(),
        rng.uniform(70e-6, 140e-6, n).tolist(),
    )
