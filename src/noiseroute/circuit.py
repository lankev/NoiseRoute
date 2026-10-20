from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp


def normalize(circuit):
    if circuit.parameters:
        raise ValueError("Bind parameters before routing")
    measured = set()
    for inst in circuit.data:
        op = inst.operation
        if (
            isinstance(op, ControlFlowOp)
            or op.name in ("reset", "delay", "store")
            or getattr(op, "condition", None)
        ):
            raise ValueError(f"Unsupported static-circuit operation: {op.name}")
        qs = {circuit.find_bit(q).index for q in inst.qubits}
        if op.name != "barrier" and qs & measured:
            raise ValueError("Only terminal measurements supported")
        if op.name == "measure":
            measured |= qs
    out = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers

    out = PassManager(RemoveBarriers()).run(out)
    out.data = [i for i in out.data if i.operation.name != "measure"] + [
        i for i in out.data if i.operation.name == "measure"
    ]
    return out


def qubit_lists(circuit):
    return [[circuit.find_bit(q).index for q in inst.qubits] for inst in circuit.data]


def materialize(circuit, device, events):
    out = QuantumCircuit(device.n, circuit.num_clbits)
    out.global_phase = circuit.global_phase
    for kind, idx, qs in events:
        if kind == "swap":
            out.swap(*qs)
        else:
            inst = circuit.data[idx]
            out.append(inst.operation, qs, [circuit.find_bit(c).index for c in inst.clbits])
    return out


def verify_trace(circuit, device, events, initial, final):
    gates = qubit_lists(circuit)
    mapping, seen, last, deps = list(initial), set(), {}, []
    for i, g in enumerate(gates):
        deps.append({last[q] for q in g if q in last})
        for q in g:
            last[q] = i
    for kind, idx, physical in events:
        if kind == "swap":
            if tuple(sorted(physical)) not in device.edges:
                raise ValueError("Off-coupling SWAP")
            a, b = physical
            mapping = [b if p == a else a if p == b else p for p in mapping]
        elif kind == "gate":
            if idx in seen or not deps[idx] <= seen or physical != [mapping[q] for q in gates[idx]]:
                raise ValueError("Invalid dependency/mapping trace")
            if len(physical) == 2 and tuple(sorted(physical)) not in device.edges:
                raise ValueError("Off-coupling gate")
            seen.add(idx)
        else:
            raise ValueError("Unknown trace event")
    if len(seen) != len(gates) or mapping != final:
        raise ValueError("Incomplete trace or wrong final permutation")
    return True
