"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD C: cross-attention peak separation (spec §5)                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Repair: as originally written, L_rep depends on argmax/NMS peak coordinates,    ║
║  which have no gradient. Here the centres are made differentiable:               ║
║    init_centers    n largest local maxima of ā (3×3 max-pool, stop-grad);        ║
║                    missing centres filled by farthest-point selection            ║
║    soft_kmeans     q_ik = softmax_k(−‖x_i − m_k‖²/τ_q),                          ║
║                    m_k ← Σ_i q_ik ā_i x_i / Σ_i q_ik ā_i, J ≈ 3 iters  Eq. (18)  ║
║    soft peak       a_k = Σ_i q_ik ā_i e^{ā_i/τ_p} / Σ_i q_ik e^{ā_i/τ_p}         ║
║  Losses                                                                          ║
║    L_act  = −(1/n) Σ_k log(a_k + ε)                                    Eq. (19)  ║
║    L_rep  = 2/(n(n−1)) Σ_{i<j} max(0, r_min − ‖m_i − m_j‖)²  (n ≥ 2)   Eq. (20)  ║
║    L_supp = Σ_{p∈S} ā(p)²,  S = local maxima farther than r_s from every m_k     ║
║                                                                        Eq. (21)  ║
║  Local maxima are found under stop-gradient, so only the values ā(p) carry       ║
║  gradient. Distances are in token units.                                         ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Provenance (§10 vii): L_act, L_rep, L_supp come from a teammate's spec. The     ║
║  soft-centre repair is proposed here. Remark 4: may find no separate peaks.      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PeakParams:
    kmeans_iters: int = 3          # J in Eq. (18)
    tau_q: float = 4.0             # assignment temperature (token units²)
    tau_p: float = 0.1             # soft-max temperature for a_k
    r_min: float = 4.0             # repulsion radius (tokens), set from shape size
    r_s: float = 3.0               # suppression radius (tokens)
    peak_min_value: float = 0.1    # local maxima used to initialise centres must exceed this
    supp_min_value: float = 0.0    # local maxima considered for L_supp must exceed this
    fps_min_value: float = 0.0     # farthest-point fill only considers tokens with ā ≥ this
    eps: float = 1e-6
    diag_peak_threshold: float = 0.3  # diagnostic only: NMS peaks above this vs n (Remark 4)


@dataclass
class PeakResult:
    l_act: torch.Tensor
    l_rep: torch.Tensor
    l_supp: torch.Tensor
    centers: torch.Tensor        # [B, M, 2] (row, col)
    center_valid: torch.Tensor   # [B, M]
    peak_values: torch.Tensor    # [B, M] soft a_k
    num_peaks: torch.Tensor      # [B] NMS peaks above diag threshold (diagnostic)


def grid_coords(h: int, w: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack([ys.flatten(), xs.flatten()], dim=-1)             # [N, 2]


@torch.no_grad()
def local_maxima(a: torch.Tensor, min_value: float) -> torch.Tensor:
    """3×3 max-pool local maxima of a [B, H, W] map, above `min_value`."""
    mp = F.max_pool2d(a[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    return (a >= mp) & (a > min_value)


@torch.no_grad()
def init_centers(a: torch.Tensor, counts: torch.Tensor, p: PeakParams) -> tuple[torch.Tensor, torch.Tensor]:
    """n largest local maxima per sample, filled by farthest-point selection (stop-grad)."""
    b, h, w = a.shape
    coords = grid_coords(h, w, a.device)
    peaks = local_maxima(a, p.peak_min_value).flatten(1)
    flat = a.flatten(1)
    m = int(counts.max().item())
    centers = torch.zeros(b, m, 2, device=a.device)
    valid = torch.zeros(b, m, dtype=torch.bool, device=a.device)
    for i in range(b):
        n = int(counts[i])
        pk = peaks[i].nonzero()[:, 0]
        order = pk[flat[i, pk].argsort(descending=True)]
        chosen = order[:n].tolist() or [int(flat[i].argmax())]
        if len(chosen) < n:
            cand = flat[i] >= p.fps_min_value
            min_d2 = ((coords[:, None, :] - coords[chosen][None]) ** 2).sum(-1).min(dim=1).values
            while len(chosen) < n:
                j = int(min_d2.masked_fill(~cand, -1.0).argmax())
                chosen.append(j)
                min_d2 = torch.minimum(min_d2, ((coords - coords[j]) ** 2).sum(-1))
        centers[i, :n] = coords[chosen]
        valid[i, :n] = True
    return centers, valid


def _assign(coords: torch.Tensor, m: torch.Tensor, valid: torch.Tensor, tau_q: float) -> torch.Tensor:
    d2 = ((coords[None, :, None, :] - m[:, None, :, :]) ** 2).sum(-1)    # [B, N, M]
    logits = (-d2 / tau_q).masked_fill(~valid[:, None, :], float("-inf"))
    return logits.softmax(dim=-1)


def soft_kmeans(a_flat: torch.Tensor, coords: torch.Tensor, centers0: torch.Tensor, valid: torch.Tensor,
                iters: int, tau_q: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (18): attention-weighted soft k-means. Returns (centres [B,M,2], assignments q [B,N,M])."""
    m = centers0
    for _ in range(iters):
        q = _assign(coords, m, valid, tau_q)
        wgt = q * a_flat[:, :, None]                                      # [B, N, M]
        mass = wgt.sum(dim=1)                                             # [B, M]
        m_new = torch.einsum("bnm,nc->bmc", wgt, coords) / (mass[..., None] + 1e-8)
        keep = valid & (mass > 1e-6)                                      # empty clusters stay put
        m = torch.where(keep[..., None], m_new, m)
    return m, _assign(coords, m, valid, tau_q)


def soft_peak_values(a_flat: torch.Tensor, q: torch.Tensor, tau_p: float) -> torch.Tensor:
    """a_k: soft maximum of ā inside cluster k (numerically stabilised exponent)."""
    e = torch.exp((a_flat - a_flat.max(dim=1, keepdim=True).values.detach()) / tau_p)
    num = torch.einsum("bnm,bn->bm", q, a_flat * e)
    den = torch.einsum("bnm,bn->bm", q, e)
    return num / (den + 1e-12)


def peak_losses(a: torch.Tensor, counts: torch.Tensor, p: PeakParams) -> PeakResult:
    """Method C error signal for one unrolled step. a: [B, H, W] ā in [0, 1] (with grad), counts: [B]."""
    b, h, w = a.shape
    coords = grid_coords(h, w, a.device)
    a_flat = a.flatten(1)
    centers0, valid = init_centers(a.detach(), counts, p)
    m, q = soft_kmeans(a_flat, coords, centers0, valid, p.kmeans_iters, p.tau_q)
    ak = soft_peak_values(a_flat, q, p.tau_p)
    n = counts.float()
    vf = valid.float()

    # Eq. (19)
    l_act = -(torch.log(ak + p.eps) * vf).sum(dim=1) / n

    # Eq. (20)
    diff = m[:, :, None, :] - m[:, None, :, :]
    dist = torch.sqrt(diff.pow(2).sum(-1) + 1e-12)                        # [B, M, M]
    pair = torch.triu(vf[:, :, None] * vf[:, None, :], diagonal=1)
    rep = (F.relu(p.r_min - dist) ** 2 * pair).sum(dim=(1, 2))
    n_pairs = (n * (n - 1) / 2).clamp_min(1.0)
    l_rep = torch.where(n >= 2, rep / n_pairs, torch.zeros_like(rep))

    # Eq. (21)
    with torch.no_grad():
        maxima = local_maxima(a, p.supp_min_value).flatten(1)            # [B, N]
        dc = torch.cdist(coords[None].expand(b, -1, -1), m.detach())    # [B, N, M]
        far = ((dc > p.r_s) | ~valid[:, None, :]).all(dim=-1)
        supp_set = (maxima & far).float()
        num_peaks = local_maxima(a, p.diag_peak_threshold).flatten(1).sum(-1)
    l_supp = (a_flat ** 2 * supp_set).sum(dim=1)

    return PeakResult(l_act=l_act.mean(), l_rep=l_rep.mean(), l_supp=l_supp.mean(), centers=m,
                      center_valid=valid, peak_values=ak, num_peaks=num_peaks)
