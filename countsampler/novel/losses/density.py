"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD B: density-map counter loss (spec §4.2)                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  L_den = Σ_k [ α ReLU(n_k − N̂_k) + β ReLU(N̂_k − n_k) ],  N̂_k = Σ_xy C_ψ(x̂0)_k ║
║                                                                     Eq. (16)    ║
║  n_k = 0 for categories the caption does not mention (penalises stray objects).  ║
║  Variants: tolerance  ReLU(|N̂ − n| − δ_c) (asymmetric α/β kept), Huber.          ║
║  Timestep weight w(t): constant | ᾱ_t | 1[t < τ] (constant for one-step Turbo).   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Provenance (§10 vi): adapted from a teammate's count-loss spec for fine-tuning   ║
║  the denoiser. Here it trains the sampler through the frozen generator.          ║
║  Remark 3: the sampler can exploit the counter. Check with an independent hard   ║
║  counter (countsampler/evaluation/hard_counters.py).                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x ** 2, delta * (ax - 0.5 * delta))


def density_count_loss(n_hat: torch.Tensor, n_target: torch.Tensor, alpha: float = 1.0, beta: float = 1.0,
                       variant: str = "hinge", tolerance: float = 0.0, huber_delta: float = 1.0) -> torch.Tensor:
    """Eq. (16) per sample. n_hat, n_target: [B, K] -> [B] (summed over categories)."""
    under = n_target - n_hat
    over = n_hat - n_target
    if variant == "hinge":
        per = alpha * F.relu(under) + beta * F.relu(over)
    elif variant == "tolerance":
        per = alpha * F.relu(under - tolerance) + beta * F.relu(over - tolerance)
    elif variant == "huber":
        per = alpha * _huber(F.relu(under), huber_delta) + beta * _huber(F.relu(over), huber_delta)
    else:
        raise ValueError("density variant must be 'hinge', 'tolerance' or 'huber'")
    return per.sum(dim=-1)


def timestep_weight(kind: str, t: int, alpha_bar: torch.Tensor, threshold: int = 500) -> torch.Tensor:
    """w(t) of §4.2: 'constant' (1), 'alpha_bar' (ᾱ_t) or 'threshold' (1[t < τ])."""
    if kind == "constant":
        return torch.ones((), device=alpha_bar.device)
    if kind == "alpha_bar":
        return alpha_bar.float()
    if kind == "threshold":
        return torch.tensor(float(t < threshold), device=alpha_bar.device)
    raise ValueError("timestep_weight must be 'constant', 'alpha_bar' or 'threshold'")


def category_targets(category_index: dict[str, int], colors: list[str], shapes: list[str],
                     counts: torch.Tensor) -> torch.Tensor:
    """n_k from the caption: the prompt's (color, shape) channel gets n, all other channels get 0."""
    target = torch.zeros(len(colors), len(category_index), device=counts.device)
    for b, (c, s) in enumerate(zip(colors, shapes)):
        target[b, category_index[f"{c} {s}"]] = counts[b].float()
    return target
