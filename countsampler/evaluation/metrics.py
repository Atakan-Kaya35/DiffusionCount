"""Count metrics, layout regularity and diversity (spec §8)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def count_metrics(pred: list[int] | np.ndarray, target: list[int] | np.ndarray) -> dict[str, float]:
    p, t = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if p.size == 0:
        return {}
    out = {"n": int(p.size), "exact_acc": float((p == t).mean()), "within1_acc": float((np.abs(p - t) <= 1).mean()),
           "mae": float(np.abs(p - t).mean()),
           "rel_err": float((np.abs(p - t) / np.maximum(t, 1)).mean())}
    per = defaultdict(list)
    for pi, ti in zip(p, t):
        per[int(ti)].append(pi == ti)
    for n, hits in sorted(per.items()):
        out[f"exact_acc@{n}"] = float(np.mean(hits))
    return out


def nn_regularity(centers: list[tuple[float, float]]) -> float | None:
    """Coefficient of variation of nearest-neighbour distances. Low values mean grid-like layouts."""
    if len(centers) < 3:
        return None
    c = np.asarray(centers)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    return float(nn.std() / (nn.mean() + 1e-9))


@torch.no_grad()
def pairwise_diversity(images: torch.Tensor, size: int = 64) -> float:
    """Mean pairwise L2 between downsampled images of the same prompt (different seeds)."""
    if images.shape[0] < 2:
        return float("nan")
    x = F.interpolate(images.float(), size=(size, size), mode="bilinear", antialias=True).flatten(1)
    d = torch.cdist(x, x) / (x.shape[1] ** 0.5)
    iu = torch.triu_indices(len(x), len(x), offset=1)
    return float(d[iu[0], iu[1]].mean())


class ClipScorer:
    """Optional prompt–image CLIP similarity (image-quality / faithfulness proxy)."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device

    @torch.no_grad()
    def __call__(self, images_uint8: np.ndarray, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, images=list(images_uint8), return_tensors="pt", padding=True)
        out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        i = F.normalize(out.image_embeds, dim=-1)
        t = F.normalize(out.text_embeds, dim=-1)
        return (100.0 * (i * t).sum(-1)).cpu().numpy()
