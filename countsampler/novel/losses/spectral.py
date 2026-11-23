"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD A: spectral instance count from self-attention (spec §3)      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  select_foreground   F: tokens with ā ≥ τ_a, top-K by ā, under stop-grad  §3.1    ║
║  attention_affinity  W_ij = (½(S_ij + S_ji))^γ on F                Eq. (10)       ║
║  feature_affinity    fallback: W_ij = max(0, cos(f_i, f_j))^γ      §3 Remark 2    ║
║  normalized_laplacian L = I − D^{-½} W D^{-½}, D = diag(W1 + ρ)    Eq. (11)       ║
║  soft_count          k̂ = Σ_i σ((τ − λ_i)/T)                        Eq. (12)       ║
║  count_loss          Huber_η(log(k̂ + ε_k) − log n)                 Eq. (13)       ║
║  gap_loss            max(0, τ_g − λ_{n+1})                          Eq. (14)       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Stabilisation (§3.3): eigvalsh in float64 on K×K (K ≤ 256), batched. Only        ║
║  eigenvalues enter the loss, so the eigh backward has no 1/(λ_i − λ_j) term and   ║
║  no custom backward is needed. Degree ridge ρ; γ and T annealed by the method.    ║
║  Provenance (§10): self-/cross-attention reading follows Make It Count (which used ║
║  DBSCAN). The spectral grouping and these losses are proposed here, untested.     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Open choices exposed here (spec §9 says they are set by the offline scan):
  * ridge_mode="relative": ρ = ridge · mean foreground degree (stop-grad), so ρ does not depend
    on the raw size of attention probabilities. "absolute" uses ρ = ridge directly.
  * drop_self_loops: the spec keeps W_ii. Dropping it stops weakly connected single tokens
    from counting as separate components.
  * min_foreground: samples with |F| < max(min_foreground, n+1) are masked out of the loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SpectralParams:
    K: int = 256
    tau_a: float = 0.3
    gamma: float = 1.0
    tau: float = 0.1
    T: float = 0.02
    tau_g: float = 0.2
    ridge: float = 1e-3
    ridge_mode: str = "relative"
    drop_self_loops: bool = False
    eps_k: float = 0.1
    huber_eta: float = 0.5
    min_foreground: int = 8


@dataclass
class SpectralResult:
    l_cnt: torch.Tensor        # scalar (masked mean over valid samples)
    l_gap: torch.Tensor        # scalar
    k_hat: torch.Tensor        # [B] soft count (float64)
    eigvals: torch.Tensor      # [B, K] ascending (float64)
    lambda_next: torch.Tensor  # [B] λ_{n+1}
    n_fg: torch.Tensor         # [B] foreground size |F|
    valid: torch.Tensor        # [B] bool


# ------------------------------------------------------------------------------- §3.1
@torch.no_grad()
def select_foreground(a_bar_flat: torch.Tensor, K: int, tau_a: float) -> tuple[torch.Tensor, torch.Tensor]:
    """F = {i : ā_i ≥ τ_a}, capped at the top K by ā. Stop-gradient. Returns (idx [B,K], mask [B,K])."""
    k = min(K, a_bar_flat.shape[-1])
    vals, idx = a_bar_flat.topk(k, dim=-1)
    return idx, vals >= tau_a


def gather_submatrix(S: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """S[b][idx[b]][:, idx[b]] for a batch. S: [B, N, N], idx: [B, K] -> [B, K, K]."""
    b, n, _ = S.shape
    k = idx.shape[1]
    rows = S.gather(1, idx[:, :, None].expand(b, k, n))
    return rows.gather(2, idx[:, None, :].expand(b, k, k))


def _mask_affinity(W: torch.Tensor, mask: torch.Tensor, drop_self_loops: bool) -> torch.Tensor:
    m = mask.to(W.dtype)
    W = W * m[:, :, None] * m[:, None, :]
    if drop_self_loops:
        W = W * (1.0 - torch.eye(W.shape[-1], device=W.device, dtype=W.dtype))
    return W


# ------------------------------------------------------------------------------- Eq. (10)
def attention_affinity(S_F: torch.Tensor, mask: torch.Tensor, gamma: float,
                       drop_self_loops: bool = False) -> torch.Tensor:
    """W_ij = (½(S_ij + S_ji))^γ on the foreground, float64. Masked tokens get zero rows/cols."""
    S_F = S_F.double()
    W = (0.5 * (S_F + S_F.transpose(1, 2))).clamp_min(0.0).pow(gamma)
    return _mask_affinity(W, mask, drop_self_loops)


def feature_affinity(feats_F: torch.Tensor, mask: torch.Tensor, gamma: float,
                     drop_self_loops: bool = False) -> torch.Tensor:
    """Fallback variant (Remark 2): W_ij = max(0, cos(f_i, f_j))^γ."""
    f = F.normalize(feats_F.double(), dim=-1)
    W = (f @ f.transpose(1, 2)).clamp_min(0.0).pow(gamma)
    return _mask_affinity(W, mask, drop_self_loops)


# ------------------------------------------------------------------------------- Eq. (11)
def normalized_laplacian(W: torch.Tensor, mask: torch.Tensor, ridge: float,
                         ridge_mode: str = "relative") -> torch.Tensor:
    """L = I − D^{-½} W D^{-½} with D = diag(W1 + ρ). Masked tokens become isolated (eigenvalue 1)."""
    deg = W.sum(dim=-1)
    if ridge_mode == "relative":
        m = mask.to(W.dtype)
        mean_deg = (deg * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        rho = (ridge * mean_deg).detach().clamp_min(1e-30)[:, None]
    elif ridge_mode == "absolute":
        rho = ridge
    else:
        raise ValueError("ridge_mode must be 'relative' or 'absolute'")
    d_is = (deg + rho).rsqrt()
    eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
    L = eye - d_is[:, :, None] * W * d_is[:, None, :]
    return 0.5 * (L + L.transpose(1, 2))


# ------------------------------------------------------------------------------- Eq. (12)-(14)
def soft_count(eigvals: torch.Tensor, tau: float, T: float) -> torch.Tensor:
    """k̂ = Σ_i σ((τ − λ_i)/T): a smooth count of near-zero eigenvalues (= number of groups)."""
    return torch.sigmoid((tau - eigvals) / T).sum(dim=-1)


def huber(x: torch.Tensor, eta: float) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax <= eta, 0.5 * x ** 2, eta * (ax - 0.5 * eta))


def count_loss(k_hat: torch.Tensor, n: torch.Tensor, eps_k: float, eta: float) -> torch.Tensor:
    """Eq. (13): Huber_η(log(k̂ + ε_k) − log n), per sample."""
    return huber(torch.log(k_hat + eps_k) - torch.log(n.to(k_hat.dtype)), eta)


def gap_loss(eigvals: torch.Tensor, n: torch.Tensor, tau_g: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (14): max(0, τ_g − λ_{n+1}). λ is ascending and 0-indexed, so λ_{n+1} = eigvals[:, n]."""
    idx = n.long().clamp_max(eigvals.shape[-1] - 1)
    lam_next = eigvals.gather(1, idx[:, None])[:, 0]
    return F.relu(tau_g - lam_next), lam_next


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.to(x.dtype)
    return (x * v).sum() / v.sum().clamp_min(1.0)


def spectral_losses_from_affinity(W: torch.Tensor, mask: torch.Tensor, n: torch.Tensor,
                                  p: SpectralParams) -> SpectralResult:
    L = normalized_laplacian(W, mask, p.ridge, p.ridge_mode)
    eigvals = torch.linalg.eigvalsh(L)                                   # float64, ascending
    k_hat = soft_count(eigvals, p.tau, p.T)
    l_cnt = count_loss(k_hat, n, p.eps_k, p.huber_eta)
    l_gap, lam_next = gap_loss(eigvals, n, p.tau_g)
    n_fg = mask.sum(-1)
    valid = (n_fg >= p.min_foreground) & (n_fg > n)                      # need K > n (§3.2)
    return SpectralResult(l_cnt=_masked_mean(l_cnt, valid).float(), l_gap=_masked_mean(l_gap, valid).float(),
                          k_hat=k_hat, eigvals=eigvals, lambda_next=lam_next, n_fg=n_fg, valid=valid)


def spectral_count_losses(S: torch.Tensor, a_bar_flat: torch.Tensor, n: torch.Tensor, p: SpectralParams,
                          feats: torch.Tensor | None = None) -> SpectralResult:
    """Full Method A error signal for one unrolled step.

    S: [B, N, N] head-averaged self-attention probabilities (rows sum to one).
    a_bar_flat: [B, N] object-token cross-attention ā (only defines F, no gradient).
    n: [B] target counts. feats: [B, N, C] if the feature-cosine fallback is used.
    """
    idx, mask = select_foreground(a_bar_flat.detach(), p.K, p.tau_a)
    if feats is None:
        W = attention_affinity(gather_submatrix(S, idx), mask, p.gamma, p.drop_self_loops)
    else:
        f_F = feats.gather(1, idx[:, :, None].expand(-1, -1, feats.shape[-1]))
        W = feature_affinity(f_F, mask, p.gamma, p.drop_self_loops)
    return spectral_losses_from_affinity(W, mask, n, p)
