# Method B: density-map counter loss (spec §4)

Code: `countsampler/novel/components/counter.py` (counter), `countsampler/novel/losses/density.py` (loss),
`countsampler/novel/methods/method_b.py` (objective), `countsampler/training/counter_trainer.py` (counter training).
Config: `configs/counter.yaml`, `configs/method_b.yaml`.

## Idea

This is the only method with a **grounded, external** count signal. A small density-map counter C_ψ is trained once on
synthetic shapes and then frozen. During sampler training it reads the one-step clean estimate x̂0, and the sampler
is trained so that the counter's soft counts match the prompt. It is an amortised version of D2D, with a trained counter
where D2D uses a detector.

## The counter C_ψ (§4.1)

**Architecture** (`DensityCounter`): a fully convolutional encoder–decoder (conv–GN–SiLU blocks, average-pool
downsampling, bilinear upsampling with skip connections, `channel_mult` [1, 2, 4] × 64). It has K output channels,
one non-negative density map per (colour, shape) category, at the input resolution. About 1.9 M parameters for K = 48.
The soft count is N̂_k = Σ_{x,y} D_k(x, y).

- **Non-negativity:** softplus with the head bias initialised at −10. softplus(0) ≈ 0.69 summed over 64×64 pixels would read as about 2800 objects in every empty channel at initialisation. With −10 it reads about 0.2.
- **Input space** (`counter.input_space`): `latent` (default, as the spec prefers) reads the scaled VAE latent x̂0 directly. `image` decodes x̂0 with the VAE, with gradient, and resizes to `counter.image_size`. This is much heavier.
- **Categories:** `data.colors × (data.train_shapes + data.heldout_shapes)`, e.g. 6 × 8 = 48.

**Training data** (`data/counter_data.py`, `data/shapes.py`):
- Synthetic scenes with 1–3 categories, each with 1–10 objects (at most 20 in total), 5 % empty scenes, light random backgrounds, object radius 4.5–9 % of the image. Half the scenes place shapes touching each other (probability 0.3 per shape), and 30 % use identical size and rotation per category.
- **Targets:** one 2-D Gaussian per object centre in its category's channel, σ = 0.025 × output side (1.6 latent pixels), normalised to sum to exactly 1. Each channel therefore sums to the object count.
- **Held-out combinations:** 10 % of (category, count) pairs never appear in training. Half of the validation scenes contain one.
- **Latent mode:** scenes are VAE-encoded once into `data_cache/counter/` (the file name hashes the relevant config, so a changed config never reuses a stale cache). Image-space augmentation is applied before encoding.
- **Optional harvested data:** `scripts/harvest_generator_samples.py` generates x̂0 with the frozen model for every category and count, labels each image with the hard counter, and saves it in the same cache format. Add it with `counter.dataset.harvested`. §4.1 suggests this because x̂0 is blurrier than clean renders.

**Loss:**

```math
\mathcal L_{C_\psi} = \operatorname{MSE}(s\,D,\ s\,\text{target}) + \lambda_{L1}\,\big|\textstyle\sum_{xy} D_k - n_k\big|
```

with s = `counter.train.mse_scale` (100) and λ_L1 = `counter.train.l1_weight` (0.1). Both the prediction and the target are scaled,
so soft counts still sum to n (see [spec_decisions.md](../spec_decisions.md)).

**Augmentation** ("heavy", §4.1): contrast, Gaussian blur and Gaussian noise. They are applied once in image space before encoding
(`counter.image_aug`) and again on every training batch (`counter.online_aug`). Latents are also flipped at random.

**Validation and freezing:** after each epoch the trainer logs count MAE and exact accuracy overall, on present
categories and on held-out combinations. `counter_best.pt` is the epoch with the lowest `val/mae_present`.
`load_counter` returns the model in `eval()` with `requires_grad=False`. It is never added to the optimiser and is
**not** wrapped in `no_grad`, so gradients flow through it to x̂0.

## The loss (§4.2)

With n_k from the caption, and **n_k = 0 for every category not mentioned** (this penalises stray objects of the wrong type):

```math
\mathcal L_{\text{den}} = \sum_{k=1}^{K} \Big[\alpha\,\mathrm{ReLU}(n_k - \hat N_k) + \beta\,\mathrm{ReLU}(\hat N_k - n_k)\Big],
\qquad \hat N_k = \sum_{x,y} \big[C_\psi(\hat x_0)\big]_k \qquad \text{(Eq. 16)}
```

| `method.density.variant` | Per-category term |
|---|---|
| `hinge` (default) | α ReLU(n − N̂) + β ReLU(N̂ − n) |
| `tolerance` | α ReLU(n − N̂ − δ_c) + β ReLU(N̂ − n − δ_c). Equals ReLU(\|N̂ − n\| − δ_c) when α = β, and reduces oscillation |
| `huber` | α Huber(ReLU(n − N̂)) + β Huber(ReLU(N̂ − n)) |

The timestep weight w(t) (`method.density.timestep_weight`) is `constant`, `alpha_bar` (ᾱ_t) or `threshold` (1[t < τ]).
For one-step Turbo models t is fixed, so w is a constant.

**Schedule (Eq. 17):** stage 1 {den, anc, pri} → stage 2 (step 3000) adds ord. No attention maps are used, so the
generator runs without capture processors.

## Knobs

| Key | Default | Notes |
|---|---|---|
| `method.density.counter_ckpt` | `runs/counter/counter_best.pt` | trained counter. Its categories must cover every colour × shape in `data` |
| `method.density.alpha`, `beta` | 1.0, 1.0 | under- and over-count weights (OPEN). Raise β to punish stray objects harder |
| `method.density.variant`, `tolerance`, `huber_delta` | hinge, 0.25, 1.0 | see the table above |
| `method.density.timestep_weight`, `timestep_threshold` | constant, 500 | w(t) |
| `counter.*` | see [configuration.md](../configuration.md) | counter architecture, data and training |

## Diagnostics (log keys)

| Key | Meaning |
|---|---|
| `B/count_abs_err` | \|N̂_prompted − n\|: what the counter thinks of the prompted category |
| `B/count_exact` | share with round(N̂_prompted) = n |
| `B/stray_mass` | mean summed N̂ over all other categories (stray objects) |

## Risks (Remark 3)

**Counter exploitation.** The sampler can learn noises whose x̂0 has the right density sum without the right image.
The counter's own numbers will then look perfect. Always judge Method B with the **independent hard counter**
(`scripts/evaluate.py`) and the image-quality and diversity metrics. Mitigations from the spec:
- stronger counter augmentation (`counter.online_aug`, `counter.image_aug`);
- harvested generator samples (`counter.dataset.harvested`);
- an **ensemble of counters**. This is not implemented yet. To add it, load several checkpoints in `DensityCountMethod.__init__` and average their `count(...)` outputs in `method_terms`.

The synthetic-to-real gap is the other risk. Pre-check B measures how well the counter reads the frozen generator's x̂0.

## How to modify

- **Replace the counter** (e.g. a pretrained differentiable counter for real images, §4.1): it only needs
  `count(x) -> [B, K]` and the category list. Pass it to `DensityCountMethod(cfg, generator, counter=..., counter_meta=...)`.
- **Change the loss:** edit `density_count_loss` or add a variant.
- **Train on decoded images:** `--set counter.input_space=image` for the counter. Method B then reads the counter's
  `input_space` from its checkpoint and decodes x̂0 with gradient. Expect much higher memory use.
