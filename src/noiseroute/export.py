import json
import time
from pathlib import Path

import numpy as np

from .benchmark import provenance
from .datasets import builtin
from .device import synthetic
from .env import RoutingEnv, Shape


def export_onnx(checkpoint, output):
    import onnx
    import onnxruntime as ort
    import torch

    from .model import INPUTS, load_model, tensor_obs

    model, shape, metadata = load_model(checkpoint)
    torch.set_num_threads(1)
    env = RoutingEnv(builtin("random", 5, 42, 2), synthetic("line", 5), shape=shape, exact_terminal=False)
    inputs = tensor_obs(env.observe())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(inputs[k] for k in INPUTS),
        str(output),
        input_names=INPUTS,
        output_names=["logits", "value"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    for _ in range(12):
        obs = env.observe()
        with torch.no_grad():
            expected = model(**tensor_obs(obs))[0].numpy()
        actual = session.run(None, {x.name: obs[x.name][None] for x in session.get_inputs()})[0]
        error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, atol=2e-4, rtol=2e-4):
            raise ValueError(f"ONNX parity failure: {error}")
        errors.append(error)
        if env.completed:
            break
        env.step(int(actual.argmax()))
    report = dict(
        shape=shape.__dict__,
        inputs=[x.name for x in session.get_inputs()],
        max_abs_error=max(errors),
        checkpoint_metadata=metadata,
        target="static batch=1; NPU compilation must be tested on real target",
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


class RuntimePolicy:
    def __init__(self, path, engine="onnx", device="CPU"):
        self.engine, self.path = engine, str(path)
        self.shape = Shape(**json.loads(Path(path).with_suffix(".json").read_text())["shape"])
        if engine == "onnx":
            import onnxruntime as ort

            if device != "CPU":
                raise ValueError("ONNX adapter is CPU only; use OpenVINO for NPU")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(self.path, options, providers=["CPUExecutionProvider"])
        elif engine == "openvino":
            import openvino as ov

            core = ov.Core()
            if device not in core.available_devices:
                raise RuntimeError(
                    f"{device} unavailable; detected {core.available_devices}. No silent fallback."
                )
            self.session = core.compile_model(
                core.read_model(self.path), device, {"PERFORMANCE_HINT": "LATENCY"}
            )
        else:
            raise ValueError(engine)

    def __call__(self, obs):
        if self.engine == "onnx":
            feed = {x.name: obs[x.name][None] for x in self.session.get_inputs()}
            return self.session.run(["logits"], feed)[0][0]
        feed = {x.get_any_name(): obs[x.get_any_name()][None] for x in self.session.inputs}
        return self.session(feed)[self.session.output(0)][0]


def latency(path, engine="onnx", device="CPU", iterations=200, output=None):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    started = time.perf_counter()
    policy = RuntimePolicy(path, engine, device)
    load_ms = (time.perf_counter() - started) * 1000
    env = RoutingEnv(
        builtin("random", 5, 7, 2), synthetic("line", 5), shape=policy.shape, exact_terminal=False
    )
    obs = env.observe()
    for _ in range(20):
        policy(obs)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        policy(obs)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    result = dict(
        engine=engine,
        device=device,
        iterations=iterations,
        model_load_compile_ms=load_ms,
        inference_median_ms=float(np.median(timings)),
        inference_p95_ms=float(np.percentile(timings, 95)),
        scope="one fixed 5-qubit placement observation padded to bucket; no feature construction",
        shape=policy.shape.__dict__,
        environment=provenance(),
    )
    from .router import route

    _, info = route(env.circuit, env.device, policy=policy, shape=policy.shape)
    result.update(sample_end_to_end_route_ms=info["route_ms"], sample_fallback_steps=info["fallback_steps"])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
    return result
