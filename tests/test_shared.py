import math

import torch

from countsampler.novel.shared import (
    CountNoiseSampler,
    GeometryLosses,
    alpha_scale,
    g_max,
    normalized_distance,
)


def test_sampler_starts_at_identity():
    s = CountNoiseSampler(latent_channels=4, text_dim=32, base_channels=16, channel_mult=(1, 2))
    eps = torch.randn(3, 4, 16, 16)
    z = s(eps, torch.tensor([1, 5, 10]), torch.randn(3, 32))
    assert torch.equal(z, eps)


def test_sampler_is_count_dependent_after_update():
    s = CountNoiseSampler(latent_channels=4, text_dim=32, base_channels=16, channel_mult=(1, 2), w=1.0)
    torch.nn.init.normal_(s.conv_out.weight, std=0.1)
    eps = torch.randn(1, 4, 16, 16).repeat(2, 1, 1, 1)
    c = torch.randn(1, 32).repeat(2, 1)
    z = s(eps, torch.tensor([1, 7]), c)
    assert not torch.allclose(z[0], z[1])


def test_alpha_makes_ordering_feasible():
    for kind in ("log", "lin"):
        a = alpha_scale(0.3, kind, 10)
        assert math.isclose(a * g_max(kind, 10), 0.6)


def test_anchor_and_prior_zero_cases():
    geo = GeometryLosses(delta=0.3)
    eps = torch.randn(2, 4, 8, 8)
    z = eps[:, None].repeat(1, 3, 1, 1, 1)
    assert geo.anchor_loss(z, eps).item() == 0.0
    unit = torch.randn(2, 3, 4, 8, 8)
    unit = unit / unit.flatten(2).pow(2).mean(-1).sqrt()[..., None, None, None]
    assert geo.prior_loss(unit).item() < 1e-10


def test_collinear_embedding_satisfies_ordering_exactly():
    """Remark 1: z_n = ε + h(n)u satisfies L_ord exactly, so it is only a smoothness prior."""
    geo = GeometryLosses(delta=0.3, g="log", n_max=10)
    eps = torch.randn(1, 4, 8, 8)
    u = torch.randn(4, 8, 8)
    u = u / u.flatten().norm() * math.sqrt(u.numel())    # ‖u‖/√D = 1
    counts = torch.tensor([[1, 3, 7, 10]])
    z = torch.stack([eps[0] + geo.alpha * math.log(n) * u for n in counts[0].tolist()])[None]
    assert geo.ordering_loss(z, counts).item() < 1e-8


def test_ranking_loss_and_finite_grads_at_identity():
    geo = GeometryLosses(delta=0.3, ord_form="rank", rank_margin=0.05)
    eps = torch.randn(2, 4, 8, 8)
    z = eps[:, None].repeat(1, 4, 1, 1, 1).requires_grad_(True)
    counts = torch.tensor([[1, 2, 5, 9], [2, 3, 4, 10]])
    loss = geo.order_loss(z, counts) + geo.anchor_loss(z, eps) + geo.prior_loss(z)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert normalized_distance(z[:, 0], z[:, 1]).max().item() < 1e-5
