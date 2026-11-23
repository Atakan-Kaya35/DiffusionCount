# Shared components: sampler and geometry losses (spec §2)

Code: `countsampler/novel/shared.py`, deliberately one file. Config blocks: `sampler:` and `geometry:` in `configs/base.yaml`.
Every method (A, B, C) uses everything on this page.

## The sampler Δφ (§2.1)

```math
z_n = \varepsilon + w\,\Delta_\phi(\varepsilon, n, \bar c) \qquad \text{(Eq. 2)}
```

- **ε**: the base noise, shared by all counts of one training example (§9: "same ε for all counts").
- **n**: the requested count, given as a sinusoidal encoding of log n.
- **c̄**: the pooled prompt embedding (CLIP pooled output for SD-Turbo, text_encoder_2 projection for SDXL-Turbo).
- **w**: a fixed mixing weight (`sampler.w`, OPEN in the spec).

### Architecture (`CountNoiseSampler`)

A small conv U-Net with FiLM conditioning, about **3.2 M parameters**:

```
ε  [4×64×64] ─ conv_in ─► 64ch @64×64 ── FiLMResBlock ─────────────────────────┐ skip
                           └ stride-2 conv ─► @32×32 ── FiLMResBlock (→128) ─┐ │ skip
                                              └ stride-2 conv ─► @16×16 ── FiLMResBlock (→128)
                                                       mid: FiLMResBlock + global self-attention (4 heads)
              decoder mirrors the encoder: nearest ×2 upsample + conv, concat skip, FiLMResBlock
GroupNorm → SiLU → conv_out (zero-initialised) ─► Δ [4×64×64]
```

Conditioning vector (`CountNoiseSampler.condition`):

```
[ sin/cos(log n · f_k), 32 frequencies log-spaced 1…64 ]  (64 dims)
[ Linear(LayerNorm(c̄)) ]                                   (128 dims)   ← dropped if sampler.use_text=false
          → MLP (SiLU) → 256-dim condition
```

Each `FiLMResBlock` does GN → SiLU → conv → GN·(1 + γ) + β → SiLU → conv, plus a residual connection, with (γ, β) = Linear(condition).
The global self-attention at 16×16 lets the sampler reason about the whole layout, which matters for placing n objects.

**Zero initialisation.** `conv_out` starts at zero, so at step 0 z_n = ε for every n and the sampler is the identity.
Training starts exactly at the frozen baseline. As in ControlNet, only `conv_out` receives a gradient on the first step.

### Knobs

| Key | Default | Effect |
|---|---|---|
| `sampler.w` | 1.0 | scales Δ. With the anchor loss active, larger w mostly changes the effective learning rate |
| `sampler.base_channels`, `sampler.channel_mult` | 64, [1,2,2] | width and depth (levels) |
| `sampler.blocks_per_level` | 1 | FiLM blocks per resolution |
| `sampler.mid_attention` | true | global self-attention at the lowest resolution |
| `sampler.use_text` | true | condition on c̄ as well as n. With `false`, the sampler sees only n |

## Geometry losses (§2.2)

Distances are normalised by √D (D = 4·64·64 = 16384), so ‖ε‖/√D ≈ 1 and **δ = 0.3 means "change the noise by up to
30 % of its norm"**:

```math
d(a,b) = \frac{\lVert a-b\rVert_2}{\sqrt D} \qquad \text{(Eq. 3)}
```

The target distance between the noises of counts n and m is α·g(n, m), with a switchable g (`geometry.g`):

```math
g_{\log}(n,m) = |\log n - \log m|, \qquad g_{\text{lin}}(n,m) = \frac{|n-m|}{N_{\max}} \qquad \text{(Eq. 4)}
```

| Loss | Formula | Code | Purpose |
|---|---|---|---|
| L_ord (regression) | $\mathbb E_\varepsilon \sum_{n<m} (d(z_n,z_m) - \alpha g(n,m))^2$ (Eq. 5) | `GeometryLosses.ordering_loss` | noises of nearby counts stay close, of distant counts far apart |
| L_ord (ranking) | $\mathbb E \max(0, d(z_a,z_b) - d(z_a,z_c) + \mu)$ for $g(a,b) < g(a,c)$ (Eq. 6) | `GeometryLosses.ranking_loss` | same ordering, without fixing the scale |
| L_anc | $\mathbb E_\varepsilon \sum_n \max(0, d(z_n,\varepsilon) - \delta)^2$ (Eq. 7) | `GeometryLosses.anchor_loss` | every z_n stays within a δ-ball around ε |
| L_pri | $\mathbb E_\varepsilon \sum_n (\lVert z_n\rVert^2/D - 1)^2$ (Eq. 9) | `GeometryLosses.prior_loss` | keep the noise at the scale the frozen model expects |

**Feasibility (Eq. 8).** All z_n lie within δ of ε, so pairwise distances are at most 2δ. The scale is therefore fixed as
α = 2δ / g_max, with g_max = log N_max (log) or (N_max − 1)/N_max (lin). For δ = 0.3 and N_max = 10:
α = 0.261 (log) or 0.667 (lin). This is `alpha_scale` in the code.

**Which pairs.** With `geometry.over: sampled` (the default, §7 step 5), the losses run over the J counts drawn for each ε.
With `all_counts`, the sampler is also evaluated for every n = 1…N_max (cheap: no generator involved) and Eq. 5 is
applied literally.

**Remark 1.** L_ord alone is solved by z_n = ε + h(n)·u for any fixed direction u, which carries no count information
(`tests/test_shared.py::test_collinear_embedding_satisfies_ordering_exactly` checks this). L_ord is a smoothness
regulariser. It comes in at the last stage, and the count signal must come from Method A, B or C.
At initialisation all z_n are equal, so L_ord has zero gradient. This is another reason it only enters at a later stage.
The distance uses a NaN-safe square root, so this case is harmless.

### Knobs

| Key | Default | Effect |
|---|---|---|
| `geometry.delta` | 0.3 | radius of the δ-ball (OPEN). Larger values allow stronger edits and risk leaving the noise distribution |
| `geometry.g` | log | `log` or `lin`. The spec asks for separate runs of each |
| `geometry.ord_form` | regression | `regression` (Eq. 5) or `rank` (Eq. 6) |
| `geometry.rank_margin` | 0.05 | μ in Eq. 6 |
| `geometry.over` | sampled | `sampled` or `all_counts` |

Loss weights live with each method: `method.weights.anc`, `.pri`, `.ord`.

## Diagnostics

Logged every step (see [outputs.md](../outputs.md)):

| Key | Meaning | Healthy |
|---|---|---|
| `geom/d_anchor_mean`, `geom/d_anchor_max` | d(z_n, ε) | ≤ δ. If max sits above δ, L_anc is fighting the method loss: raise `anc` |
| `geom/norm_sq` | mean ‖z‖²/D | ≈ 1. Raise `pri` if it drifts |
| `geom/d_pair_mean` | mean d(z_n, z_m) | rises from 0 as the sampler learns count-dependent edits |
| `loss/anc`, `loss/pri`, `loss/ord` | the losses themselves (logged even when inactive) | |

## How to modify

- **Different conditioning** (e.g. a learned embedding of n): change `sinusoidal_count_encoding` / `CountNoiseSampler.condition`.
- **Different architecture**: anything that maps (ε, n, c̄) to a tensor shaped like ε works. Keep a zero-initialised last layer so training starts at the baseline.
- **A new geometry loss**: add a method to `GeometryLosses`, compute it in `CountMethod.compute` (`novel/methods/base.py`),
  add `"<name>"` to `CountMethod.SHARED_TERMS`, and give it a weight and a stage in the method configs.
