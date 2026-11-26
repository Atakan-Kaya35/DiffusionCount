

def test_norm_preserving_sampler_keeps_z_on_the_shell():
    """The projection that lets Dphi reach the displacement count control needs.

    Learned additive directions were measured to collapse by a normalised distance of 0.08,
    while slerp paths survive to 1.42 - the difference being that slerp holds ||z|| fixed. With
    norm_preserve the sampler can take an arbitrarily large step and still land on the shell,
    so the budget that made the two incompatible no longer binds.
    """
    import torch
    from countsampler.novel.shared import CountNoiseSampler

    torch.manual_seed(0)
    eps = torch.randn(4, 4, 16, 16)
    n = torch.randint(1, 11, (4,))
    c_bar = torch.randn(4, 32)
    common = dict(latent_channels=4, text_dim=32, base_channels=8, channel_mult=(1, 2),
                  blocks_per_level=1, mid_attention=False, cond_dim=16, count_embed_dim=8,
                  text_embed_dim=8, groups=4)

    plain = CountNoiseSampler(**common, w=50.0, norm_preserve=False)
    # conv_out is zero-initialised so the sampler starts as the exact identity; give it a real
    # delta, otherwise both forms trivially return eps and the test proves nothing
    with torch.no_grad():
        plain.conv_out.weight.normal_(0.0, 0.1)
    z_plain = plain(eps, n, c_bar)
    proj = CountNoiseSampler(**common, w=50.0, norm_preserve=True)
    proj.load_state_dict(plain.state_dict())
    z_proj = proj(eps, n, c_bar)

    en = eps.flatten(1).norm(dim=1)
    # a large w drives the additive form off the shell; the projection puts it back exactly
    assert (z_plain.flatten(1).norm(dim=1) - en).abs().max() > 1e-2
    assert torch.allclose(z_proj.flatten(1).norm(dim=1), en, rtol=1e-4, atol=1e-4)
    # and it is still a real displacement, not a no-op that returns eps
    d = (z_proj - eps).flatten(1).norm(dim=1) / (eps[0].numel() ** 0.5)
    assert (d > 0.1).all()
