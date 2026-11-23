import pytest
import torch
from conftest import TINY_CROSS, TINY_SELF

from countsampler.generator import AttentionRequest, object_attention_map


def test_layer_scan(tiny_gen):
    info = tiny_gen.attention_layers()
    assert info[TINY_SELF].kind == "self" and info[TINY_SELF].grid == (16, 16)
    assert info[TINY_CROSS].kind == "cross"
    assert any(li.grid == (8, 8) for li in info.values())


def test_capture_processor_matches_default(tiny_gen):
    z = torch.randn(2, *tiny_gen.latent_shape)
    enc = tiny_gen.encode_prompts(["three red circles", "one blue square"])
    with torch.no_grad():
        ref = tiny_gen.forward_at(z, 999, enc).eps_hat
        tiny_gen.watch(AttentionRequest([TINY_SELF], [TINY_CROSS], capture_features=True))
        out = tiny_gen.forward_at(z, 999, enc, capture=True)
        tiny_gen.unwatch()
    assert torch.allclose(ref, out.eps_hat, atol=1e-5)
    S = out.self_attn[TINY_SELF]
    assert S.shape == (2, 256, 256) and torch.allclose(S.sum(-1), torch.ones(2, 256), atol=1e-5)
    assert out.cross_attn[TINY_CROSS].shape == (2, 256, 77)
    assert out.self_feats[TINY_SELF].shape[:2] == (2, 256)


def test_clean_estimate_formula(tiny_gen):
    z = torch.randn(1, *tiny_gen.latent_shape)
    enc = tiny_gen.encode_prompts(["two green stars"])
    with torch.no_grad():
        out = tiny_gen.forward_at(z, 999, enc)
    ab = tiny_gen.alpha_bar(999)
    assert torch.allclose(out.x0_hat, (z - (1 - ab).sqrt() * out.eps_hat) / ab.sqrt(), atol=1e-5)
    outs = tiny_gen.unroll(z, enc, [999, 499])
    assert len(outs) == 2 and outs[1].t == 499


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_unroll_matches_diffusers_euler_trailing(tiny_gen, steps):
    """With input_scaling='pipeline' our unroll is exactly SD-Turbo's Euler/trailing pipeline."""
    from diffusers import EulerDiscreteScheduler

    sch = EulerDiscreteScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                                 timestep_spacing="trailing")
    sch.set_timesteps(steps)
    z = torch.randn(2, *tiny_gen.latent_shape)
    enc = tiny_gen.encode_prompts(["three red circles", "one blue square"])
    with torch.no_grad():
        latents = z * sch.init_noise_sigma
        for t in sch.timesteps:
            eps = tiny_gen._model_output(sch.scale_model_input(latents, t), int(t), enc)
            latents = sch.step(eps, t, latents).prev_sample
        ours = tiny_gen.unroll(z, enc, [int(t) for t in sch.timesteps])[-1].x0_hat
    assert int(sch.timesteps[0]) == 999
    assert torch.allclose(latents, ours, atol=1e-3, rtol=1e-4)


def test_gradients_reach_noise_through_attention_with_checkpointing(tiny_gen):
    tiny_gen.watch(AttentionRequest([TINY_SELF], [TINY_CROSS]))
    try:
        z = torch.randn(2, *tiny_gen.latent_shape, requires_grad=True)
        enc = tiny_gen.encode_prompts(["three red circles", "five red circles"])
        out = tiny_gen.unroll(z, enc, capture=True)[0]
        tok = tiny_gen.token_indices(["three red circles", "five red circles"], ["circles", "circles"])
        a = object_attention_map(out.cross_attn, [TINY_CROSS], tok, tiny_gen.layer_grids(), (16, 16), 0.5)
        loss = out.self_attn[TINY_SELF].pow(2).sum() + a.pow(2).sum() + out.x0_hat.pow(2).mean()
        loss.backward()
    finally:
        tiny_gen.unwatch()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny_gen.unet.parameters())


def test_token_indices(tiny_gen):
    idx = tiny_gen.token_indices(["a photo of three red circles"], ["circles"])[0]
    assert idx == [6]  # BOS + 5 words before it
