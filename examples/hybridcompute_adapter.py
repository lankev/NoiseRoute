"""Minimal callable boundary for a host dispatcher; no external HybridCompute API assumed."""

from noiseroute.export import RuntimePolicy


def make_routing_policy(model_path, accelerator="CPU"):
    """Return a callable observation -> logits, with .shape for RoutingEnv.

    A host such as HybridCompute can own model loading and select CPU/NPU.
    This is an integration boundary, not a tested adapter to an unseen host API.
    """
    return RuntimePolicy(model_path, engine="openvino", device=accelerator)
