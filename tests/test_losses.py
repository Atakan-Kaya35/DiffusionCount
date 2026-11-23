import math

import torch

from countsampler.novel.components import DensityCounter, freeze_counter
from countsampler.novel.losses import (
    PeakParams,
    SpectralParams,
    category_targets,
    density_count_loss,
    peak_losses,
    spectral_count_losses,
)


# ----------------------------------------------------------------------------- Method A
def _block_attention(n_tokens: int, groups: int, leak: float = 1e-4) -> torch.Tensor:
    """Row-stochastic self-attention with `groups` instance blocks (tokens attend within their block)."""
    labels = torch.arange(n_tokens) % groups
    S = (labels[:, None] == labels[None, :]).float() + leak
    return (S / S.sum(-1, keepdim=True))[None]


def test_soft_count_recovers_number_of_groups():
    p = SpectralParams(K=60, tau_a=0.0, gamma=1.0, tau=0.1, T=0.01, min_foreground=2)
    for r in (1, 3, 5):
        S = _block_attention(60, r)
        res = spectral_count_losses(S, torch.ones(1, 60), torch.tensor([r]), p)
        assert abs(res.k_hat.item() - r) < 0.2, (r, res.k_hat)
        assert res.l_cnt.item() < 1e-2
        assert res.l_gap.item() == 0.0


def test_gap_fires_when_too_many_groups():
    p = SpectralParams(K=60, tau_a=0.0, tau_g=0.2, min_foreground=2)
    res = spectral_count_losses(_block_attention(60, 5), torch.ones(1, 60), torch.tensor([2]), p)
    assert res.l_gap.item() > 0.1
    assert res.l_cnt.item() > 0.1


def test_masked_background_does_not_count():
    p = SpectralParams(K=60, tau_a=0.5, T=0.01, min_foreground=2)
    a = torch.zeros(1, 60)
    a[0, :30] = 1.0          # only 30 foreground tokens
    S = _block_attention(60, 2)
    res = spectral_count_losses(S, a, torch.tensor([2]), p)
    assert int(res.n_fg) == 30
    assert abs(res.k_hat.item() - 2) < 0.2


def test_spectral_gradient_reaches_attention_logits():
    logits = torch.randn(2, 64, 64, requires_grad=True)
    S = logits.softmax(-1)
    for affinity_feats in (None, torch.randn(2, 64, 16, requires_grad=True)):
        p = SpectralParams(K=48, tau_a=0.0, gamma=2.0, min_foreground=2)
        res = spectral_count_losses(S, torch.rand(2, 64), torch.tensor([2, 4]), p, feats=affinity_feats)
        (res.l_cnt + res.l_gap).backward(retain_graph=True)
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


# ----------------------------------------------------------------------------- Method C
def _bumps(h, w, centers, sigma=1.5):
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    a = sum(torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2)) for cy, cx in centers)
    return (a / a.max())[None]


def test_peak_centers_converge_to_bumps():
    a = _bumps(24, 24, [(4, 4), (4, 19), (19, 12)]).requires_grad_(True)
    p = PeakParams(r_min=3.0, r_s=3.0)
    res = peak_losses(a, torch.tensor([3]), p)
    got = sorted(tuple(round(v) for v in c.tolist()) for c in res.centers[0])
    assert got == [(4, 4), (4, 19), (19, 12)]
    assert res.l_rep.item() == 0.0 and res.l_supp.item() < 1e-6
    assert res.l_act.item() < 0.2               # soft peak ≈ 0.9 (soft max over the cluster)
    (res.l_act + res.l_rep + res.l_supp).backward()
    assert torch.isfinite(a.grad).all()


def test_extra_peaks_are_suppressed_and_missing_ones_filled():
    a = _bumps(24, 24, [(4, 4), (4, 19), (19, 12)])
    res2 = peak_losses(a, torch.tensor([2]), PeakParams(r_s=3.0))
    assert res2.l_supp.item() > 0.5            # the third bump is an extra peak
    a4 = a.clone().requires_grad_(True)
    res4 = peak_losses(a4, torch.tensor([4]), PeakParams())
    assert res4.center_valid.sum() == 4         # farthest-point fill for the missing centre
    (res4.l_act + res4.l_rep).backward()
    assert torch.isfinite(a4.grad).all()


# ----------------------------------------------------------------------------- Method B
def test_density_loss_variants():
    n_hat = torch.tensor([[2.5, 0.4]])
    tgt = torch.tensor([[3.0, 0.0]])
    assert math.isclose(density_count_loss(n_hat, tgt, 1.0, 2.0).item(), 0.5 + 0.8, rel_tol=1e-6)
    assert math.isclose(density_count_loss(n_hat, tgt, variant="tolerance", tolerance=0.45).item(), 0.05,
                        rel_tol=1e-5)
    assert density_count_loss(n_hat, tgt, variant="huber").item() > 0


def test_counter_init_is_near_zero_and_frozen_counter_passes_gradients():
    c = DensityCounter(in_channels=4, num_categories=6, base_channels=8, channel_mult=(1, 2))
    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    assert c.count(x).max().item() < 0.5        # strongly negative head bias
    freeze_counter(c)
    c.count(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in c.parameters())


def test_category_targets_zero_for_unmentioned():
    idx = {"red circle": 0, "red square": 1, "blue circle": 2}
    t = category_targets(idx, ["red", "blue"], ["square", "circle"], torch.tensor([3, 5]))
    assert t.tolist() == [[0, 3, 0], [0, 0, 5]]
