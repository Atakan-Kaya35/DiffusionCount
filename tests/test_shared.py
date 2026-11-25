import math

import pytest
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


# ---------------------------------------------------------------- output anchor (L_x0anc)
def _geo(**kw):
    from countsampler.novel.shared import GeometryLosses
    return GeometryLosses(**kw)


def test_x0_anchor_is_zero_inside_the_budget():
    """Nothing is penalised while the output stays within delta_x of the frozen generation."""
    g = _geo(delta_x=0.15)
    ref = torch.randn(2 * 3, 4, 8, 8)
    x0 = ref.clone()                       # d = 0
    assert float(g.x0_anchor_loss(x0, ref, be=2)) == pytest.approx(0.0)


def test_x0_anchor_penalises_only_the_excess():
    """The hinge measures how far PAST delta_x the output moved, not how far it moved."""
    from countsampler.novel.shared import normalized_distance
    g = _geo(delta_x=0.10)
    ref = torch.zeros(1 * 2, 4, 8, 8)
    x0 = torch.zeros_like(ref)
    # put sample 0 at distance 0.30 and sample 1 at 0.05 (inside the budget)
    for i, s in enumerate((0.30, 0.05)):
        v = torch.randn(4, 8, 8)
        x0[i] = v / v.flatten().norm() * s * (4 * 8 * 8) ** 0.5
    d = normalized_distance(x0, ref)
    assert d[0] == pytest.approx(0.30, abs=1e-4)
    assert d[1] == pytest.approx(0.05, abs=1e-4)
    # only sample 0 is outside: (0.30 - 0.10)^2 = 0.04
    assert float(g.x0_anchor_loss(x0, ref, be=1)) == pytest.approx(0.04, abs=1e-5)


def test_x0_anchor_sums_over_counts_and_means_over_eps():
    """Same reduction as L_anc: sum over the J counts, mean over the base noises."""
    g = _geo(delta_x=0.0)
    be, j = 3, 2
    ref = torch.zeros(be * j, 4, 4, 4)
    x0 = torch.zeros_like(ref)
    for i in range(be * j):
        v = torch.randn(4, 4, 4)
        x0[i] = v / v.flatten().norm() * 0.5 * (4 * 4 * 4) ** 0.5
    # every sample is at d = 0.5 with delta_x = 0 -> each contributes 0.25, summed over j
    assert float(g.x0_anchor_loss(x0, ref, be=be)) == pytest.approx(0.25 * j, abs=1e-5)


def test_x0_diagnostics_report_distance_not_loss():
    g = _geo(delta_x=0.15)
    ref = torch.zeros(4, 4, 4, 4)
    x0 = torch.zeros_like(ref)
    v = torch.randn(4, 4, 4)
    x0[0] = v / v.flatten().norm() * 0.4 * (4 * 4 * 4) ** 0.5
    d = g.x0_diagnostics(x0, ref)
    assert d["geom/d_x0_max"] == pytest.approx(0.4, abs=1e-4)
    assert d["geom/d_x0_mean"] == pytest.approx(0.1, abs=1e-4)


# ---------------------------------------------------------------- g = pow (measured spacing)
def test_g_pow_matches_the_measured_count_spacing():
    """The generator's own x0 geometry gives d(1,2)/d(7,8) = 1.65; p = 0.7 reproduces it."""
    from countsampler.novel.shared import g_pow
    n = torch.tensor([1, 7]); m = torch.tensor([2, 8])
    v = g_pow(n, m, 0.7)
    assert float(v[0] / v[1]) == pytest.approx(1.63, abs=0.05)


def test_g_pow_sits_between_lin_and_log():
    """log is too steep (5.19) and lin too flat (1.00); pow has to land between them."""
    from countsampler.novel.shared import g_lin, g_log, g_pow
    n = torch.tensor([1, 7]); m = torch.tensor([2, 8])
    r_lin = float(g_lin(n, m, 10)[0] / g_lin(n, m, 10)[1])
    r_log = float(g_log(n, m)[0] / g_log(n, m)[1])
    r_pow = float(g_pow(n, m, 0.7)[0] / g_pow(n, m, 0.7)[1])
    assert r_lin == pytest.approx(1.0, abs=1e-6)
    assert r_log > 5.0
    assert r_lin < r_pow < r_log


def test_g_pow_is_symmetric_and_zero_on_the_diagonal():
    from countsampler.novel.shared import g_pow
    n = torch.tensor([3, 5]); m = torch.tensor([5, 3])
    assert float(g_pow(n, m, 0.7)[0]) == pytest.approx(float(g_pow(n, m, 0.7)[1]))
    same = torch.tensor([4])
    assert float(g_pow(same, same, 0.7)[0]) == pytest.approx(0.0, abs=1e-7)


def test_alpha_scale_keeps_pow_targets_inside_the_delta_ball():
    """alpha * g_max must equal 2*delta for every metric, so all pairs fit the anchor budget."""
    from countsampler.novel.shared import GeometryLosses, g_pow
    for kind in ("log", "lin", "pow"):
        G = GeometryLosses(delta=0.3, g=kind, n_max=10)
        n = torch.tensor([1]); m = torch.tensor([10])
        assert float(G.alpha * G.g(n, m)[0]) == pytest.approx(0.6, abs=1e-5)
    G = GeometryLosses(delta=0.3, g="pow", n_max=10, g_power=0.7)
    assert float(G.g(torch.tensor([1]), torch.tensor([10]))[0]) == pytest.approx(10 ** 0.7 - 1, abs=1e-5)


def test_geometry_rejects_an_unknown_metric():
    from countsampler.novel.shared import GeometryLosses
    with pytest.raises(ValueError, match="geometry.g"):
        GeometryLosses(g="sqrt")
