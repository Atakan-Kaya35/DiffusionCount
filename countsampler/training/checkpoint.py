"""Sampler checkpoint I/O (plain dicts and tensors, so it loads with weights_only=True)."""
from __future__ import annotations

from pathlib import Path

import torch

from ..novel.shared import CountNoiseSampler, build_sampler
from ..utils.config import to_dict, wrap


def save_sampler_checkpoint(path: str | Path, sampler: CountNoiseSampler, cfg, step: int, *,
                            latent_channels: int, text_dim: int, optimizer=None, lr_scheduler=None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": sampler.state_dict(),
        "sampler_config": to_dict(cfg.sampler),
        "latent_channels": int(latent_channels),
        "text_dim": int(text_dim),
        "step": int(step),
        "config": to_dict(cfg),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)


def load_sampler_checkpoint(path: str | Path, device="cpu") -> tuple[CountNoiseSampler, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sampler = build_sampler(wrap(ckpt["sampler_config"]), ckpt["latent_channels"], ckpt["text_dim"])
    sampler.load_state_dict(ckpt["state_dict"])
    return sampler.to(device).eval(), ckpt
