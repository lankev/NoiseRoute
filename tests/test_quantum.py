import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator

from noiseroute.benchmark import compile_one
from noiseroute.datasets import builtin
from noiseroute.device import synthetic
from noiseroute.env import RoutingEnv
from noiseroute.metrics import evaluate, hellinger
from noiseroute.router import route


def assert_equivalent(qc, routed, info, n):
    restored = routed.copy()
    current = info["final"].copy()
    for q, desired in enumerate(info["initial"]):
        actual = current[q]
        if actual != desired:
            restored.swap(actual, desired)
            current = [desired if p == actual else actual if p == desired else p for p in current]
    expected = QuantumCircuit(n)
    expected.compose(qc, qubits=info["initial"][: qc.num_qubits], inplace=True)
    assert Operator(restored).equiv(Operator(expected))


@pytest.mark.parametrize("kind", ["line", "ring", "grid", "neutral"])
@pytest.mark.parametrize("family", ["random", "qft", "grover", "arithmetic", "ansatz"])
def test_routing_equivalence(kind, family):
    d = synthetic(kind, 5)
    qc = builtin(family, 4, 31, 2)
    routed, info = route(qc, d)
    assert_equivalent(qc, routed, info, d.n)
    assert sorted(info["final"]) == list(range(d.n))
    assert sum(x.operation.name == "swap" for x in routed.data) == info["swaps"]
    for inst in routed.data:
        if len(inst.qubits) == 2:
            assert tuple(sorted(routed.find_bit(q).index for q in inst.qubits)) in d.edges


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_plugin_levels(level):
    qc = builtin("random", 5, 15, 4)
    d = synthetic("line", 5)
    compiled = transpile(
        qc,
        target=d.target(),
        layout_method="noiseroute",
        routing_method="noiseroute",
        optimization_level=level,
        seed_transpiler=14,
    )
    assert Operator.from_circuit(compiled).equiv(Operator(qc))
    assert all(i.operation.name != "swap" for i in compiled.data)


def test_measurement_and_registers():
    from qiskit_aer import AerSimulator

    qr, cr = QuantumRegister(4, "logic"), ClassicalRegister(4, "bits")
    qc = QuantumCircuit(qr, cr)
    qc.x(0)
    qc.cx(0, 3)
    qc.cx(3, 1)
    for q, c in enumerate([3, 1, 0, 2]):
        qc.measure(q, c)
    out, _ = compile_one(qc, synthetic("line", 5), "noiseroute")
    sim = AerSimulator()
    expected = sim.run(transpile(qc, sim), shots=20).result().get_counts()
    assert sim.run(out, shots=20).result().get_counts() == expected


def test_masks_and_failure():
    env = RoutingEnv(builtin("qft", 5), synthetic("line", 5), max_steps=1)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError):
        env.step(env.shape.nodes)
    _, reward, done, truncated, info = env.step(0)
    assert done and not truncated and info["failed"] and reward <= -10000
    with pytest.raises(ValueError):
        env.output()


def test_esp_analytical():
    d = synthetic("line", 2)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    m = evaluate(qc, d)
    assert m["esp"] == pytest.approx(1 - d.cal[("cx", (0, 1))]["error"])
    assert m["log_idle"] == 0
    qc.cx(0, 1)
    assert evaluate(qc, d)["esp"] < m["esp"]
    assert hellinger({"0": 1}, {"1": 1}) == 0
    assert hellinger({"0": 1, "1": 1}, {"0": 0.5, "1": 0.5}) == pytest.approx(1)


def test_unsupported_inputs():
    for kind in ("reset", "delay", "mid_measure"):
        qc = QuantumCircuit(2, 2)
        if kind == "reset":
            qc.reset(0)
        elif kind == "delay":
            qc.delay(2, 0)
        else:
            qc.measure(0, 0)
            qc.x(0)
        with pytest.raises(ValueError):
            RoutingEnv(qc, synthetic("line", 2))


def test_beam_and_empty():
    d = synthetic("ring", 4)
    q = builtin("random", 4, 23, 2)
    out, info = route(q, d, beam=2)
    assert_equivalent(q, out, info, 4)
    empty = QuantumCircuit(4)
    out, info = route(empty, d)
    assert_equivalent(empty, out, info, 4)
