"""Small shared helpers: seeding, device/dtype resolution, schedules, blur, image conversion."""
from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype '{name}'") from exc


def scheduled_value(spec: Any, step: int) -> float:
    """A scalar, or a linear anneal {start, end, steps[, begin]} evaluated at `step`."""
    if isinstance(spec, (int, float)):
        return float(spec)
    start, end = float(spec["start"]), float(spec["end"])
    begin = int(spec.get("begin", 0))
    steps = max(1, int(spec["steps"]))
    frac = min(1.0, max(0.0, (step - begin) / steps))
    return start + frac * (end - start)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a [B, C, H, W] tensor (differentiable, replicate padding)."""
    if sigma is None or sigma <= 0:
        return x
    c = x.shape[1]
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)
    return x


def to_uint8_images(images: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor [B, 3, H, W] -> uint8 numpy [B, H, W, 3]."""
    x = ((images.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int, pad: int = 4) -> None:
    """Save [-1, 1] images [B, 3, H, W] as a grid with `nrow` images per row."""
    from PIL import Image

    arr = to_uint8_images(images)
    b, h, w, _ = arr.shape
    ncol = nrow
    nrows = math.ceil(b / ncol)
    canvas = np.full((nrows * (h + pad) + pad, ncol * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i in range(b):
        r, c = divmod(i, ncol)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = arr[i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
