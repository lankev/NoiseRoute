"""Static dense GATv2 + DAG transformer with shared placement/routing actor."""

from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn

from .env import Shape


class GATv2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.left, self.right = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.att = nn.Linear(dim, 1, bias=False)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj, mask):
        pair = torch.nn.functional.leaky_relu(self.left(x)[:, :, None, :] + self.right(x)[:, None, :, :], 0.2)
        scores = self.att(pair).squeeze(-1) + torch.log(adj.clamp_min(1e-6))
        valid = (adj > 0) & (mask[:, None, :] > 0)
        valid = valid | ((mask[:, :, None] == 0) & (mask[:, None, :] > 0))
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
        return self.norm(x + weights @ self.value(x)) * mask[:, :, None]


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.dim = heads, dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x, mask):
        b, length, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, length, 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        scores = q @ k.transpose(-1, -2) / (self.dim // self.heads) ** 0.5
        probs = torch.softmax(scores.masked_fill(mask[:, None, None, :] == 0, -1e4), dim=-1)
        y = (probs @ v).transpose(1, 2).reshape(b, length, self.dim)
        x = self.norm1(x + self.proj(y))
        return self.norm2(x + self.ff(x)) * mask[:, :, None]


class JointPolicy(nn.Module):
    def __init__(self, dim=64, layers=2, hierarchical=True):
        super().__init__()
        self.config = dict(dim=dim, layers=layers, hierarchical=hierarchical)
        self.hierarchical = hierarchical
        self.node_in, self.token_in = nn.Linear(8, dim), nn.Linear(8, dim)
        self.graph = nn.ModuleList([GATv2(dim) for _ in range(layers)])
        self.transformer = nn.ModuleList([AttentionBlock(dim) for _ in range(layers)])
        self.coarse = GATv2(dim)
        self.place = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.swap = nn.Sequential(nn.Linear(dim * 3 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.critic = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, nodes, adj, tokens, edges, node_mask, token_mask, action_mask, phase, clusters):
        x, t = self.node_in(nodes), self.token_in(tokens)
        for graph, transformer in zip(self.graph, self.transformer):
            x, t = graph(x, adj, node_mask), transformer(t, token_mask)
        if self.hierarchical:
            count = clusters.sum(dim=1).clamp_min(1)
            cx = clusters.transpose(1, 2) @ x / count[:, :, None]
            ca = (clusters.transpose(1, 2) @ adj @ clusters).clamp(0, 1)
            cm = (clusters.sum(dim=1) > 0).to(x.dtype)
            x = x + clusters @ self.coarse(cx, ca, cm)
        context = t.sum(1) / token_mask.sum(1, keepdim=True).clamp_min(1)
        graph_context = (x * node_mask[:, :, None]).sum(1) / node_mask.sum(1, keepdim=True).clamp_min(1)
        place = self.place(
            torch.cat(
                [
                    x,
                    context[:, None, :].expand(-1, x.shape[1], -1),
                    phase[:, None, :].expand(-1, x.shape[1], -1),
                ],
                -1,
            )
        ).squeeze(-1)
        a = torch.gather(x, 1, edges[:, :, 0, None].expand(-1, -1, x.shape[-1]))
        b = torch.gather(x, 1, edges[:, :, 1, None].expand(-1, -1, x.shape[-1]))
        swaps = self.swap(
            torch.cat(
                [
                    a + b,
                    torch.abs(a - b),
                    context[:, None, :].expand(-1, a.shape[1], -1),
                    phase[:, None, :].expand(-1, a.shape[1], -1),
                ],
                -1,
            )
        ).squeeze(-1)
        logits = torch.cat([place, swaps], -1).masked_fill(action_mask == 0, -1e4)
        value = self.critic(torch.cat([graph_context, context, phase], -1)).squeeze(-1)
        return logits, value


INPUTS = ["nodes", "adj", "tokens", "edges", "node_mask", "token_mask", "action_mask", "phase", "clusters"]


def tensor_obs(obs, device="cpu", batch=True):
    return {
        k: torch.as_tensor(v, device=device).unsqueeze(0) if batch else torch.as_tensor(v, device=device)
        for k, v in obs.items()
    }


def save_model(path, model, shape, metadata=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        state_dict=model.state_dict(),
        config=model.config,
        shape=asdict(shape),
        metadata=metadata or {},
        format_version=1,
    )
    if training_state is not None:
        payload["training_state"] = training_state
    # Keep the preceding checkpoint intact if saving is interrupted.
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=True)
    model = JointPolicy(**obj["config"]).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, Shape(**obj["shape"]), obj.get("metadata", {})


class TorchPolicy:
    def __init__(self, path, device="cpu"):
        self.model, self.shape, self.metadata = load_model(path, device)
        self.device = device

    def __call__(self, obs):
        with torch.inference_mode():
            logits, _ = self.model(**tensor_obs(obs, self.device))
        return logits[0].cpu().numpy()
