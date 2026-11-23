"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  COMPONENT: density-map counter C_ψ (spec §4.1, Method B)             ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  A small fully convolutional encoder–decoder with K output channels: one         ║
║  non-negative density map D_k per (colour, shape) category, all from one forward ║
║  pass. Soft count N̂_k = Σ_xy D_k(x, y).                                          ║
║  softplus output with a strongly negative head bias, because softplus(0) ≈ 0.69  ║
║  summed over H·W would give a large spurious count in empty channels at init.    ║
║  Input space: VAE latents (preferred, reads x̂0 directly) or decoded RGB images.  ║
║  Trained once (countsampler/training/counter_trainer.py), then frozen:           ║
║  eval(), requires_grad=False, not in the optimiser, NOT under no_grad.           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.GroupNorm(min(groups, ch_out), ch_out), nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1), nn.GroupNorm(min(groups, ch_out), ch_out), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensityCounter(nn.Module):
    def __init__(self, in_channels: int, num_categories: int, base_channels: int = 64,
                 channel_mult=(1, 2, 4), out_activation: str = "softplus", head_bias_init: float = -10.0,
                 groups: int = 8):
        super().__init__()
        chs = [base_channels * m for m in channel_mult]
        self.out_activation = out_activation
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.enc.append(ConvBlock(prev, c, groups))
            prev = c
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.dec.append(ConvBlock(prev + chs[i], chs[i], groups))
            prev = chs[i]
        self.head = nn.Conv2d(prev, num_categories, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, head_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, H, W] -> density [B, K, H, W] ≥ 0 (same resolution as the input)."""
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = F.avg_pool2d(h, 2)
        for blk in self.dec:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        out = self.head(h)
        if self.out_activation == "softplus":
            return F.softplus(out)
        if self.out_activation == "relu":
            return F.relu(out)
        raise ValueError("out_activation must be 'softplus' or 'relu'")

    def count(self, x: torch.Tensor) -> torch.Tensor:
        """Soft counts N̂_k = Σ_xy D_k(x, y). [B, K]."""
        return self(x).sum(dim=(-2, -1))


def build_counter(meta: dict) -> DensityCounter:
    arch = meta.get("arch", {})
    return DensityCounter(
        in_channels=int(meta["in_channels"]),
        num_categories=len(meta["categories"]),
        base_channels=int(arch.get("base_channels", 64)),
        channel_mult=tuple(arch.get("channel_mult", (1, 2, 4))),
        out_activation=arch.get("out_activation", "softplus"),
        head_bias_init=float(arch.get("head_bias_init", -10.0)),
    )


def freeze_counter(model: nn.Module) -> nn.Module:
    """Frozen per §4.1: eval, requires_grad=False. Callers must NOT wrap its use in no_grad."""
    model.eval()
    model.requires_grad_(False)
    return model


def save_counter(path: str | Path, model: DensityCounter, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_counter(path: str | Path, device="cpu") -> tuple[DensityCounter, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_counter(ckpt["meta"])
    model.load_state_dict(ckpt["state_dict"])
    return freeze_counter(model.to(device)), ckpt["meta"]
