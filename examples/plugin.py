from qiskit import QuantumCircuit, transpile

from noiseroute.device import synthetic

device = synthetic("ring", 5, seed=7)
qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 4)
qc.cx(0, 2)
qc.measure_all()
routed = transpile(
    qc, target=device.target(), layout_method="noiseroute", routing_method="noiseroute", optimization_level=3
)
print(routed)
print("Layout:", routed.layout)
