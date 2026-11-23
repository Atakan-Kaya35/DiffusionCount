"""Attention capture for the frozen U-Net (infrastructure, spec §1 and §3.1).

Fused attention kernels (SDPA / flash) do not expose the probability matrix, so at the
*watched* layers we swap in `CaptureAttnProcessor`, which computes
softmax(QKᵀ/√d) explicitly (in float32) and records the head-averaged probabilities.
All other layers keep their fast default processor.

The processor computes the same thing whether or not the store is recording. That matters
because gradient checkpointing re-runs the forward pass during backward, and the
recomputation has to save the same tensors as the original pass.

Captured tensors stay in the autograd graph, so losses on attention maps (Methods A and C)
send gradients back through the frozen U-Net to the noise z_n and then to the sampler φ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..utils.misc import gaussian_blur


@dataclass
class AttentionRequest:
    """Which attention layers a method needs (full module names inside the U-Net)."""

    self_layers: list[str] = field(default_factory=list)
    cross_layers: list[str] = field(default_factory=list)
    capture_features: bool = False  # self-attention output features (cosine-affinity fallback, §3)

    def is_empty(self) -> bool:
        return not (self.self_layers or self.cross_layers)


@dataclass
class LayerInfo:
    name: str
    kind: str              # "self" | "cross"
    tokens: int            # number of spatial query tokens
    grid: tuple[int, int]  # (h, w) token grid


class AttentionStore:
    """Holds captured maps for a single U-Net forward pass."""

    def __init__(self) -> None:
        self.active = False
        self._clear()

    def _clear(self) -> None:
        self.self_attn: dict[str, torch.Tensor] = {}
        self.cross_attn: dict[str, torch.Tensor] = {}
        self.self_feats: dict[str, torch.Tensor] = {}

    def begin(self, active: bool) -> None:
        self._clear()
        self.active = bool(active)

    def end(self) -> dict[str, dict[str, torch.Tensor]]:
        out = {"self_attn": self.self_attn, "cross_attn": self.cross_attn, "self_feats": self.self_feats}
        self.active = False
        self._clear()
        return out

    def record(self, name: str, kind: str, probs_mean: torch.Tensor, feats: torch.Tensor | None) -> None:
        if kind == "self":
            self.self_attn[name] = probs_mean
            if feats is not None:
                self.self_feats[name] = feats
        else:
            self.cross_attn[name] = probs_mean


def is_attention_module(module: torch.nn.Module) -> bool:
    return hasattr(module, "set_processor") and hasattr(module, "to_q") and hasattr(module, "to_k")


class CaptureAttnProcessor:
    """Drop-in replacement for diffusers' `AttnProcessor2_0` with explicit attention probabilities."""

    def __init__(self, store: AttentionStore, name: str, capture_features: bool = False):
        self.store = store
        self.name = name
        self.capture_features = capture_features

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", None):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = split_heads(query), split_heads(key), split_heads(value)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        # Explicit S = softmax(QKᵀ/√d) in float32 (spec §3.1).
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        probs = scores.softmax(dim=-1)                                  # [B, heads, Nq, Nk]
        out = torch.matmul(probs.to(value.dtype), value)                # [B, heads, Nq, hd]
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        if self.store.active:
            feats = out if (self.capture_features and not is_cross) else None
            self.store.record(self.name, "cross" if is_cross else "self", probs.mean(dim=1), feats)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if getattr(attn, "residual_connection", False):
            out = out + residual
        return out / attn.rescale_output_factor


def install_capture_processors(unet: torch.nn.Module, store: AttentionStore,
                               request: AttentionRequest) -> dict[str, object]:
    """Install capture processors on the requested layers. Returns the original processors."""
    modules = dict(unet.named_modules())
    wanted = list(dict.fromkeys(list(request.self_layers) + list(request.cross_layers)))
    missing = [n for n in wanted if n not in modules or not is_attention_module(modules[n])]
    if missing:
        raise KeyError(f"attention layers not found in U-Net: {missing}. Run scripts/list_layers.py.")
    originals = {}
    for name in wanted:
        module = modules[name]
        originals[name] = module.processor
        module.set_processor(CaptureAttnProcessor(
            store, name, capture_features=request.capture_features and name in request.self_layers))
    return originals


def restore_processors(unet: torch.nn.Module, originals: dict[str, object]) -> None:
    modules = dict(unet.named_modules())
    for name, proc in originals.items():
        modules[name].set_processor(proc)


# ----------------------------------------------------------------------------------------
# Reading captured maps
# ----------------------------------------------------------------------------------------

def minmax_normalize(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalisation of [B, H, W] maps to [0, 1]."""
    flat = a.flatten(1)
    lo = flat.min(dim=1).values[:, None, None]
    hi = flat.max(dim=1).values[:, None, None]
    return (a - lo) / (hi - lo + eps)


def token_weight_matrix(token_indices: list[list[int]], seq_len: int, device) -> torch.Tensor:
    """[B, L] averaging weights over each prompt's object sub-tokens."""
    w = torch.zeros(len(token_indices), seq_len, device=device)
    for b, idx in enumerate(token_indices):
        w[b, idx] = 1.0 / len(idx)
    return w


def object_attention_map(cross_attn: dict[str, torch.Tensor], layers: list[str],
                         token_indices: list[list[int]], layer_grids: dict[str, tuple[int, int]],
                         out_grid: tuple[int, int], smoothing_sigma: float = 0.0,
                         normalize: bool = True) -> torch.Tensor:
    """ā (spec §3.1, §5): the object token's cross-attention map.

    Head-averaged (done at capture), averaged over `layers` after resizing to `out_grid`,
    optionally lightly smoothed, then min-max normalised per sample. Returns [B, H, W].
    """
    maps = []
    for name in layers:
        probs = cross_attn[name]                                       # [B, N, L]
        w = token_weight_matrix(token_indices, probs.shape[-1], probs.device)
        m = torch.einsum("bnl,bl->bn", probs, w)
        h, wd = layer_grids[name]
        m = m.view(-1, 1, h, wd)
        if (h, wd) != tuple(out_grid):
            m = F.interpolate(m, size=tuple(out_grid), mode="bilinear", align_corners=False)
        maps.append(m)
    a = torch.stack(maps, dim=0).mean(dim=0)
    if smoothing_sigma and smoothing_sigma > 0:
        a = gaussian_blur(a, smoothing_sigma)
    a = a[:, 0]
    return minmax_normalize(a) if normalize else a


def mean_self_attention(self_attn: dict[str, torch.Tensor], layers: list[str]) -> torch.Tensor:
    """Head-averaged self-attention S [B, N, N], averaged over `layers` (same resolution required)."""
    mats = [self_attn[n] for n in layers]
    if len({m.shape for m in mats}) != 1:
        raise ValueError("self-attention layers averaged together must share a resolution")
    return torch.stack(mats, dim=0).mean(dim=0)


def concat_self_features(self_feats: dict[str, torch.Tensor], layers: list[str]) -> torch.Tensor:
    """Per-layer L2-normalised features, concatenated, so the cosine is the mean per-layer cosine."""
    feats = [F.normalize(self_feats[n].float(), dim=-1) for n in layers]
    return torch.cat(feats, dim=-1)
