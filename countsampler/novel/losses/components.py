"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD D: differentiable connected-component count                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Motivation. Method C's terms are all invariant under MERGING two objects:       ║
║  centre positions, centre distances and soft peak values inside a cluster do not ║
║  change when two blobs fuse, so one blob hosting n soft centres satisfies L_act, ║
║  L_rep and L_supp simultaneously. Measured: L_rep was exactly 0.0 for a whole    ║
║  6000-step run that produced one object for every requested count.               ║
║                                                                                  ║
║  A count objective needs a quantity that STRICTLY DECREASES when objects merge.  ║
║  The Euler characteristic of a superlevel set is one, and it is cheap:           ║
║      chi = V - E + F   over the pixel cubical complex                            ║
║          = #components - #holes   (exact for a binary field)                     ║
║  Verified on planted Gaussians at a hard threshold: chi = k exactly for k=1..10, ║
║  and merging two blobs takes chi from 2 to 1 the moment they touch.              ║
║                                                                                  ║
║  Losses                                                                          ║
║    L_cc  = (log(1 + chi) - log(1 + n))^2    symmetric in over/under-counting     ║
║    L_bin = mean(u (1 - u))                  u -> {0,1}, so chi means what it says ║
║                                                                                  ║
║  L_bin is not cosmetic. For a UNIFORM field chi ~ HW(u - 2u^2 + u^4), which      ║
║  peaks around u ~ 0.3: a structureless haze can hit any moderate chi. That       ║
║  degenerate solution sits almost exactly at the maximum of u(1-u), so L_bin is   ║
║  what closes it. Dropping either term reopens a hole.                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ComponentParams:
    """Defaults calibrated on planted Gaussians, not guessed.

    Threshold relative to the map's MAX, not its mean: `object_attention_map` min-max normalises,
    so max == 1 is a stable reference while the mean moves with how much of the map is foreground.
    At tau_frac=0.5, temperature=0.02 the soft chi equals the true object count exactly for
    k = 1, 2, 3, 5, 8, 10. A mean-relative threshold was tried first and gave chi=3.2 for ten
    objects while a structureless haze reached 11.
    """
    tau_frac: float = 0.5       # threshold as a fraction of the map's max
    temperature: float = 0.02   # sigmoid width for the soft occupancy
    min_area: float = 3.0       # tokens per component below which the area term bites
    eps: float = 1e-6
    hard_threshold: float = 0.5   # diagnostic only: the discrete chi reported alongside


@dataclass
class ComponentResult:
    l_cc: torch.Tensor        # [B] count term
    l_bin: torch.Tensor       # [B] binarisation term
    l_area: torch.Tensor      # [B] per-component area floor
    chi: torch.Tensor         # [B] soft Euler characteristic (differentiable)
    chi_hard: torch.Tensor    # [B] discrete Euler characteristic (diagnostic, no gradient)
    occupancy: torch.Tensor   # [B] mean of u, for health logging
    area_per_component: torch.Tensor   # [B] mass / chi


def euler_characteristic(u: torch.Tensor) -> torch.Tensor:
    """chi = V - E + F for a [B, H, W] occupancy field, under 4-connectivity.

    Exact for binary u. The products are the natural multilinear extension, so the quantity stays
    differentiable in between — see the module docstring for what that costs.
    """
    v = u.sum(dim=(-2, -1))
    e = (u[..., :, :-1] * u[..., :, 1:]).sum(dim=(-2, -1)) \
        + (u[..., :-1, :] * u[..., 1:, :]).sum(dim=(-2, -1))
    f = (u[..., :-1, :-1] * u[..., :-1, 1:] * u[..., 1:, :-1] * u[..., 1:, 1:]).sum(dim=(-2, -1))
    return v - e + f


def component_losses(a: torch.Tensor, counts: torch.Tensor,
                     p: ComponentParams) -> ComponentResult:
    """a: [B, H, W] attention map in [0, 1]; counts: [B] long."""
    if a.dim() != 3:
        raise ValueError(f"expected [B, H, W] attention map, got {tuple(a.shape)}")
    tau = p.tau_frac * a.amax(dim=(-2, -1), keepdim=True)
    u = torch.sigmoid((a - tau) / p.temperature)

    chi = euler_characteristic(u)
    n = counts.float()
    # Log-ratio, so over- and under-counting cost the same. Method C's L_supp punished extra peaks
    # far harder than missing ones, and that asymmetry is what pointed it at collapse.
    l_cc = (torch.log1p(chi.clamp_min(0.0)) - torch.log1p(n)) ** 2
    l_bin = (u * (1.0 - u)).mean(dim=(-2, -1))

    # Without an area floor the cheapest way to reach chi = n is n single-token specks: at low
    # occupancy chi ~ HW*u, so a sparse field hits any moderate count with no objects in it, and
    # it does so where u(1-u) is small, so L_bin does not reach it. Measured at these settings:
    # a random speck field gives 1.09 tokens per component against 7.75 for real objects.
    mass = u.sum(dim=(-2, -1))
    area = mass / chi.clamp_min(1.0)
    l_area = (torch.relu(p.min_area - area) / p.min_area) ** 2

    with torch.no_grad():
        hard = (a > p.hard_threshold * a.amax(dim=(-2, -1), keepdim=True)).float()
        chi_hard = euler_characteristic(hard)

    return ComponentResult(l_cc=l_cc, l_bin=l_bin, l_area=l_area, chi=chi, chi_hard=chi_hard,
                           occupancy=u.mean(dim=(-2, -1)), area_per_component=area)
