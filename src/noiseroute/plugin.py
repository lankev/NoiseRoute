import os
import warnings

from qiskit.converters import dag_to_circuit
from qiskit.transpiler import AnalysisPass, Layout, PassManager, TransformationPass
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .device import Device, synthetic
from .router import noise_layout, route


def get_device(config):
    path = os.environ.get("NOISEROUTE_DEVICE")
    if path:
        device = Device.load(path)
        cmap = config.coupling_map or config.target.build_coupling_map()
        if cmap is not None and (
            device.n != cmap.size() or set(device.edges) != {tuple(sorted(e)) for e in cmap.get_edges()}
        ):
            raise ValueError("Snapshot does not match Qiskit target")
        return device
    if config.target is not None and config.target.qubit_properties is not None:
        return Device.from_target(config.target)
    cmap = config.coupling_map
    if cmap is None and config.target is not None:
        cmap = config.target.build_coupling_map()
    if cmap is None:
        raise ValueError("NoiseRoute requires a coupling map or target")
    warnings.warn(
        "Connectivity-only input: using explicitly synthetic calibration, not hardware noise.", RuntimeWarning
    )
    return synthetic("custom", cmap.size(), edges=cmap.get_edges())


def get_policy():
    path = os.environ.get("NOISEROUTE_MODEL")
    if not path:
        return None
    engine = os.environ.get("NOISEROUTE_ENGINE", "onnx" if path.endswith(".onnx") else "torch")
    if engine in ("onnx", "openvino"):
        from .export import RuntimePolicy

        return RuntimePolicy(path, engine, os.environ.get("NOISEROUTE_ACCELERATOR", "CPU"))
    if engine != "torch":
        raise ValueError(engine)
    from .model import TorchPolicy

    return TorchPolicy(path)


class NoiseLayout(AnalysisPass):
    def __init__(self, device, policy=None):
        super().__init__()
        self.device, self.policy = device, policy

    def run(self, dag):
        if self.property_set["layout"] is not None:
            return
        circuit = dag_to_circuit(dag)
        if self.policy is None:
            placement = noise_layout(circuit, self.device)
        else:
            import numpy as np

            from .env import RoutingEnv

            env = RoutingEnv(circuit, self.device, shape=self.policy.shape, exact_terminal=False)
            while env.core is None:
                env.step(int(np.argmax(self.policy(env.observe()))))
            placement = env.initial_mapping[: len(dag.qubits)]
        self.property_set["layout"] = Layout({q: placement[i] for i, q in enumerate(dag.qubits)})


class NoiseRouting(TransformationPass):
    def __init__(self, device, policy=None, beam=1):
        super().__init__()
        self.device, self.policy, self.beam = device, policy, beam

    def run(self, dag):
        if len(dag.qubits) != self.device.n:
            raise ValueError("Run layout embedding first")
        routed, info = route(
            dag_to_circuit(dag),
            self.device,
            policy=self.policy,
            joint=False,
            beam=self.beam,
            shape=self.policy.shape if self.policy else None,
        )
        layout = Layout({q: info["final"][i] for i, q in enumerate(dag.qubits)})
        old = self.property_set["final_layout"]
        self.property_set["final_layout"] = layout if old is None else old.compose(layout, dag.qubits)
        self.property_set["noiseroute_stats"] = {k: v for k, v in info.items() if k != "events"}
        out = dag.copy_empty_like()
        out.global_phase = routed.global_phase
        for inst in routed.data:
            qs = [dag.qubits[routed.find_bit(q).index] for q in inst.qubits]
            cs = [dag.clbits[routed.find_bit(c).index] for c in inst.clbits]
            out.apply_operation_back(inst.operation, qs, cs)
        return out


class NoiseRoutePlugin(PassManagerStagePlugin):
    def pass_manager(self, pass_manager_config, optimization_level=None):
        return PassManager(
            NoiseRouting(
                get_device(pass_manager_config), get_policy(), int(os.environ.get("NOISEROUTE_BEAM", "1"))
            )
        )


class NoiseLayoutPlugin(PassManagerStagePlugin):
    def pass_manager(self, pass_manager_config, optimization_level=None):
        device = get_device(pass_manager_config)
        pm = PassManager(NoiseLayout(device, get_policy()))
        pm += common.generate_embed_passmanager(device.coupling)
        return pm
