"""ESP is a survival proxy, not a quantum state fidelity."""

import math

import numpy as np


def evaluate(circuit, device):
    clocks = np.zeros(device.n)
    active = np.zeros(device.n)
    measured = set()
    log_gate = 0.0
    for inst in circuit.data:
        name = inst.operation.name
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if name == "barrier":
            clocks[list(qs)] = max(clocks[list(qs)], default=0)
            continue
        if name == "delay":
            if inst.operation.unit != "s":
                raise ValueError("Convert delays to seconds")
            clocks[qs[0]] += float(inst.operation.duration)
            continue
        if (name, qs) not in device.cal:
            raise ValueError(f"Missing native calibration: {(name, qs)}")
        if any(q in measured for q in qs):
            raise ValueError("Operation after measurement unsupported")
        cal = device.cal[(name, qs)]
        start = max(clocks[list(qs)], default=0)
        clocks[list(qs)] = start + cal["duration"]
        active[list(qs)] += cal["duration"]
        log_gate += math.log1p(-cal["error"])
        if name == "measure":
            measured.update(qs)
    makespan = float(max(clocks, default=0))
    ends = np.array(
        [clocks[q] if q in measured else (makespan if active[q] > 0 else 0) for q in range(device.n)]
    )
    idle = np.maximum(ends - active, 0)
    inv_tphi = np.maximum(1 / np.array(device.t2) - 0.5 / np.array(device.t1), 0)
    log_idle = -float(np.sum(idle * (1 / np.array(device.t1) + inv_tphi)))
    log_esp = log_gate + log_idle
    return dict(
        log_esp=log_esp,
        esp=math.exp(log_esp),
        log_gate=log_gate,
        log_idle=log_idle,
        depth_2q=circuit.depth(filter_function=lambda x: len(x.qubits) == 2),
        native_2q=sum(len(x.qubits) == 2 for x in circuit.data),
        duration_s=makespan,
        idle_s=float(idle.sum()),
    )


def hellinger(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if sp <= 0 or sq <= 0:
        raise ValueError("Empty distributions")
    return sum(math.sqrt(p.get(k, 0) / sp * q.get(k, 0) / sq) for k in keys) ** 2


def noisy_validation(logical, compiled, device, shots=8192, seed=0):
    if max(logical.num_qubits, compiled.num_qubits) > 12:
        raise ValueError("Noisy simulation capped at 12 physical qubits")
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

    if logical.count_ops().get("measure", 0):
        raise ValueError("Logical reference must be unmeasured")
    if compiled.count_ops().get("measure", 0) != logical.num_qubits:
        raise ValueError("Compile logical.measure_all() first")
    noise = NoiseModel()
    for x in device.instructions:
        e, qs, name = x["error"], x["qubits"], x["name"]
        if name == "measure":
            noise.add_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]), qs)
        elif e > 0:
            d = 2 ** len(qs)
            noise.add_quantum_error(depolarizing_error(e * d / (d - 1), len(qs)), name, qs)
    scheduled = compiled.copy_empty_like()
    clocks = np.zeros(device.n)
    for inst in compiled.data:
        qs = [compiled.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "barrier":
            start = max(clocks[qs], default=0)
            for q in qs:
                wait = start - clocks[q]
                if wait > 1e-15:
                    err = thermal_relaxation_error(device.t1[q], min(device.t2[q], 2 * device.t1[q]), wait)
                    scheduled.append(err.to_instruction(), [q])
            clocks[qs] = start
            continue
        cal = device.cal[(inst.operation.name, tuple(qs))]
        start = max(clocks[qs], default=0)
        for q in qs:
            wait = start - clocks[q]
            if wait > 1e-15:
                err = thermal_relaxation_error(device.t1[q], min(device.t2[q], 2 * device.t1[q]), wait)
                scheduled.append(err.to_instruction(), [q])
        scheduled.append(inst.operation, qs, [compiled.find_bit(c).index for c in inst.clbits])
        clocks[qs] = start + cal["duration"]
    ideal = Statevector.from_instruction(logical).probabilities_dict()
    sim = AerSimulator(noise_model=noise, method="density_matrix")
    counts = (
        sim.run(transpile(scheduled, sim, optimization_level=0), shots=shots, seed_simulator=seed)
        .result()
        .get_counts()
    )
    return dict(
        hellinger=hellinger(ideal, counts),
        shots=shots,
        noise_model="depolarizing + symmetric readout + ASAP idle T1/T2",
    )
