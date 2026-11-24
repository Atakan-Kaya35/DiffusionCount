"""Method D's count term must do the one thing Method C's terms could not: notice a merge.

Method C optimised L_act + L_rep + L_supp to convergence while producing a single object for every
requested count, because each of those terms is invariant under merging two objects. These tests
pin the property that distinguishes Method D, not merely that the code runs.
"""
import math

import pytest
import torch

from countsampler.novel.losses.components import (
    ComponentParams, component_losses, euler_characteristic,
)


def blobs(centres, h=16, sigma=1.0):
    y, x = torch.meshgrid(torch.arange(h).float(), torch.arange(h).float(), indexing="ij")
    a = torch.zeros(h, h)
    for cy, cx in centres:
        a = torch.maximum(a, torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma * sigma)))
    return a


def ring(k, h=16, r=5.0):
    return blobs([(h / 2 + r * math.sin(2 * math.pi * i / k),
                   h / 2 + r * math.cos(2 * math.pi * i / k)) for i in range(k)], h)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 8, 10])
def test_euler_characteristic_equals_the_object_count(k):
    a = ring(k) if k > 1 else blobs([(8, 8)])
    chi = euler_characteristic((a > 0.5).float()[None])
    assert float(chi) == pytest.approx(k, abs=0.01)


def test_merging_two_objects_lowers_chi_by_one():
    """The property Method C's terms lacked entirely."""
    apart = euler_characteristic((blobs([(8, 5), (8, 11)]) > 0.5).float()[None])
    merged = euler_characteristic((blobs([(8, 7.5), (8, 8.5)]) > 0.5).float()[None])
    assert float(apart) == pytest.approx(2.0, abs=0.01)
    assert float(merged) == pytest.approx(1.0, abs=0.01)


def test_the_collapsed_blob_costs_more_than_the_correct_layout():
    """Method C scored one blob at 1.18 against 0.17 for the correct layout but reached the blob
    anyway, via a barrier. Here the count term itself must separate them, and by a wide margin."""
    p = ComponentParams()
    n = torch.tensor([3])
    correct = component_losses(ring(3)[None], n, p)
    collapsed = component_losses(blobs([(8, 8)], sigma=3.0)[None], n, p)
    assert float(correct.l_cc) < float(collapsed.l_cc)
    assert float(collapsed.l_cc) - float(correct.l_cc) > 0.1


def test_the_count_term_is_symmetric_in_over_and_under_counting():
    """L_supp punished extra peaks far harder than missing ones, which is what pointed Method C at
    collapse. The log-ratio form must not repeat that."""
    p = ComponentParams()
    n = torch.tensor([4])
    under = float(component_losses(ring(2)[None], n, p).l_cc)
    over = float(component_losses(ring(8)[None], n, p).l_cc)
    assert under > 0.05 and over > 0.05
    assert 0.3 < under / over < 3.0          # same order of magnitude, neither one-sided


def test_the_structureless_haze_is_penalised():
    """A uniform field must not be able to satisfy the count term.

    With a mean-relative threshold it could: a haze reached chi=11 for n=10 at an occupancy where
    u(1-u) is small, so the binarisation term never saw it. A max-relative threshold removes the
    hole at the source — a uniform map is everywhere at its own max, so it saturates into a single
    component and the count term itself rejects it.
    """
    p = ComponentParams()
    n = torch.tensor([10])
    haze = component_losses(torch.full((1, 16, 16), 0.30), n, p)
    real = component_losses(ring(10)[None], n, p)
    assert float(haze.chi) < 2.0                      # saturates to one component, not eleven
    assert float(haze.l_cc) > 50 * float(real.l_cc)   # and the count term rejects it outright


def test_a_speck_field_cannot_buy_the_count_cheaply():
    """n single-token specks reach chi = n with no objects. The area floor is what forbids it."""
    p = ComponentParams()
    torch.manual_seed(0)
    specks = (torch.rand(1, 16, 16) < 0.04).float()
    real = ring(10)[None]
    s = component_losses(specks, torch.tensor([10]), p)
    r = component_losses(real, torch.tensor([10]), p)
    assert float(s.area_per_component) < p.min_area < float(r.area_per_component)
    assert float(s.l_area) > 0.0 and float(r.l_area) == 0.0


def test_soft_and_hard_chi_agree_on_real_maps():
    """The soft count is only meaningful where it tracks the discrete one it stands in for."""
    p = ComponentParams()
    for k in (1, 2, 3, 5, 8, 10):
        a = (ring(k) if k > 1 else blobs([(8, 8)]))[None]
        res = component_losses(a, torch.tensor([k]), p)
        assert abs(float(res.chi) - float(res.chi_hard)) < 0.5, f"k={k}"
        assert float(res.chi_hard) == pytest.approx(k, abs=0.01), f"k={k}"


def test_losses_are_differentiable_with_respect_to_the_map():
    a = ring(3)[None].clone().requires_grad_(True)
    res = component_losses(a, torch.tensor([3]), ComponentParams())
    (res.l_cc.sum() + res.l_bin.sum()).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_batch_entries_are_independent():
    p = ComponentParams()
    batch = torch.stack([ring(2), ring(5)])
    both = component_losses(batch, torch.tensor([2, 5]), p)
    one = component_losses(ring(2)[None], torch.tensor([2]), p)
    assert float(both.l_cc[0]) == pytest.approx(float(one.l_cc[0]), rel=1e-5)
