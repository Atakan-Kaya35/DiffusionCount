# Configuration reference

## How configs work

Configs are YAML files in `configs/`, loaded by `countsampler/utils/config.py`.

- **Inheritance:** a file may start with `defaults: base.yaml` (or a list of files). Parents load first and the child overrides them key by key.
- **Overlays:** pass `--config` several times. Later files win: `--config configs/method_a.yaml --config configs/gpu_8gb.yaml`.
- **Command-line overrides:** `--set key.sub=value`. Values are parsed as YAML, so `--set train.steps=200`,
  `--set "generator.timesteps=[999,499]"`, `--set data.include_count=false` and `--set train.resume=null` all work.
  Quote anything with brackets or spaces.
- **Reproducibility:** each run writes its fully resolved config to `runs/<name>/config.yaml`.
  Sampler checkpoints also store it, so `scripts/evaluate.py --ckpt ...` needs no `--config`.

| File | Purpose |
|---|---|
| `configs/base.yaml` | shared defaults for everything |
| `configs/method_a.yaml`, `configs/method_b.yaml`, `configs/method_c.yaml` | one per method (inherit base) |
| `configs/counter.yaml` | counter training for Method B (inherits base) |
| `configs/sdxl_turbo.yaml` | overlay: SDXL-Turbo with its layer names |
| `configs/gpu_8gb.yaml` | overlay: smaller micro-batches for 8 GB cards |

**OPEN** marks open choices in spec §9. Set them with the offline scan or by sweeping.
Keeping this page complete is enforced by `tests/test_docs.py`.

## experiment

| Key | Default | Description |
|---|---|---|
| `experiment.name` | `default` (per method: `method_a`, …) | run directory name under `output_dir` |
| `experiment.output_dir` | `runs` | root for run directories |
| `experiment.seed` | 0 | seeds Python, NumPy, PyTorch, the prompt sampler and the noise generator |
| `experiment.device` | `auto` | `auto`, `cuda`, `cuda:1`, `cpu` |
| `experiment.wandb` | false | also log to Weights & Biases |
| `experiment.wandb_project` | `count-sampler` | W&B project name |

## generator (frozen G, §1)

| Key | Default | Description |
|---|---|---|
| `generator.model_id` | `stabilityai/sd-turbo` | Hugging Face model id |
| `generator.family` | `auto` | `sd`, `sdxl`, or `auto` (`sdxl` if the id contains "xl") |
| `generator.variant` | `fp16` | weight-file variant to download (falls back to the default files) |
| `generator.dtype` | `bfloat16` | compute dtype of the frozen model. bf16 avoids fp16 gradient underflow |
| `generator.resolution` | 512 | image size. The latent is 4 × res/8 × res/8 |
| `generator.timesteps` | `[999]` | unrolled steps. `[999, 499]` = first two steps |
| `generator.input_scaling` | `pipeline` | first U-Net input: `pipeline` (√(1−ᾱ_T)·z, exactly the diffusers Turbo pipelines) or `identity` (z, literal Eq. 1). See [spec_decisions.md](spec_decisions.md) |
| `generator.gradient_checkpointing` | true | recompute U-Net activations in backward |
| `generator.min_alpha_bar` | 1e-4 | clamp for ᾱ_t in Eq. 1 |
| `generator.vae_id` | (unset) | optional separate VAE repo, e.g. `madebyollin/sdxl-vae-fp16-fix` for fp16 SDXL |

## data (prompts, §1 and §7)

| Key | Default | Description |
|---|---|---|
| `data.n_max` | 10 | N_max. Counts are 1…N_max |
| `data.colors` | red, green, blue, yellow, orange, purple | colours in prompts. The hard counter has hue ranges for these and black |
| `data.train_shapes` | circle, square, triangle, star, hexagon | shapes used for sampler training |
| `data.heldout_shapes` | pentagon, diamond, cross | shapes held out for testing (§7) |
| `data.templates` | three templates | prompt templates with `{count}`, `{color}`, `{shape}`. The object word is the (pluralised) shape |
| `data.count_format` | `word` | `word` ("three") or `digit` ("3") |
| `data.include_count` | true | `false` = §7 ablation: the number is removed from the prompt |

## sampler (Δφ, §2.1)

| Key | Default | Description |
|---|---|---|
| `sampler.w` | 1.0 | **OPEN.** Mixing weight w |
| `sampler.base_channels` | 64 | width of the first level |
| `sampler.channel_mult` | `[1, 2, 2]` | channel multipliers per level (one stride-2 downsample per extra level) |
| `sampler.blocks_per_level` | 1 | FiLM residual blocks per level |
| `sampler.mid_attention` | true | global self-attention at the lowest resolution |
| `sampler.cond_dim` | 256 | condition vector size |
| `sampler.count_embed_dim` | 64 | size of the sinusoidal log n encoding |
| `sampler.count_embed_max_freq` | 64.0 | highest frequency of that encoding |
| `sampler.text_embed_dim` | 128 | projection size of c̄ |
| `sampler.use_text` | true | condition on c̄ as well as n |

## geometry (§2.2)

| Key | Default | Description |
|---|---|---|
| `geometry.delta` | 0.3 | **OPEN.** δ, anchor radius in normalised distance |
| `geometry.g` | `log` | `log` or `lin` target metric |
| `geometry.ord_form` | `regression` | `regression` (Eq. 5) or `rank` (Eq. 6) |
| `geometry.rank_margin` | 0.05 | μ in Eq. 6 |
| `geometry.over` | `sampled` | `sampled` (the J sampled counts, §7) or `all_counts` (1…N_max, Eq. 5 literally) |

## train (sampler training, §7)

| Key | Default | Description |
|---|---|---|
| `train.steps` | 6000 | optimiser steps |
| `train.eps_per_batch` | 2 | base noises per micro-batch |
| `train.counts_per_eps` | 4 | J: counts per base noise, label count included. U-Net batch = `eps_per_batch × counts_per_eps` |
| `train.grad_accum` | 1 | micro-batches per optimiser step |
| `train.lr` | 1e-4 | AdamW learning rate |
| `train.weight_decay` | 0.0 | AdamW weight decay |
| `train.betas` | (unset → [0.9, 0.999]) | optional AdamW betas |
| `train.warmup_steps` | 100 | linear learning-rate warm-up |
| `train.grad_clip` | 1.0 | global gradient-norm clip |
| `train.log_every` | 10 | log interval (steps) |
| `train.ckpt_every` | 1000 | checkpoint interval (0 = only the final one) |
| `train.resume` | null | sampler checkpoint to resume from (restores optimiser, scheduler and step) |
| `train.preview.every` | 500 | preview-grid interval (0 = off) |
| `train.preview.seeds` | `[0, 1]` | seeds in preview grids |
| `train.preview.counts` | `[1, 2, 3, 4, 5, 6, 8, 10]` | columns of preview grids |
| `train.preview.prompts` | `[[red, circle], [blue, pentagon]]` | (colour, shape) pairs, one grid each |

## method (per method)

Common keys:

| Key | Default | Description |
|---|---|---|
| `method.name` | `A` / `B` / `C` | which objective (`countsampler/novel/methods/__init__.py`) |
| `method.weights` | per method | **OPEN.** λ per term, e.g. `method.weights.cnt`. Shared terms: `anc`, `pri`, `ord` |
| `method.stages` | per method | list of `{start: step, terms: [...]}`. The last stage with `start ≤ step` decides which terms get gradient weight |
| `method.loss_steps` | `all` | which unrolled steps the method loss is averaged over: `all`, `last` or a list of indices |

### Method A (`configs/method_a.yaml`)

| Key | Default | Description |
|---|---|---|
| `method.attention.self_layers` | `[up_blocks.2.attentions.0.transformer_blocks.0.attn1]` | **OPEN** (layer ℓ). Self-attention layer(s) for S. Several must share a resolution |
| `method.attention.cross_layers` | `[...attn2]` of the same block | cross-attention layer(s) for ā (defines F only) |
| `method.attention.smoothing_sigma` | 0.5 | Gaussian smoothing of ā, in tokens |
| `method.spectral.affinity` | `attention` | `attention` (Eq. 10) or `feature_cosine` (Remark 2 fallback) |
| `method.spectral.K` | 256 | **OPEN.** Foreground cap |
| `method.spectral.tau_a` | 0.3 | **OPEN.** ā threshold for F |
| `method.spectral.gamma` | `{start: 1, end: 2, steps: 3000}` | **OPEN.** γ, a scalar or a linear anneal `{start, end, steps[, begin]}` |
| `method.spectral.T` | `{start: 0.05, end: 0.02, steps: 3000}` | **OPEN.** Soft-count temperature, scalar or anneal |
| `method.spectral.tau` | 0.1 | **OPEN.** Near-zero eigenvalue threshold τ |
| `method.spectral.tau_g` | 0.2 | **OPEN.** Gap threshold τ_g |
| `method.spectral.ridge` | 1e-3 | **OPEN.** Degree ridge ρ |
| `method.spectral.ridge_mode` | `relative` | `relative` (ρ × mean foreground degree) or `absolute` |
| `method.spectral.drop_self_loops` | false | zero W_ii |
| `method.spectral.eps_k` | 0.1 | ε_k in Eq. 13 |
| `method.spectral.huber_eta` | 0.5 | η in Eq. 13 |
| `method.spectral.min_foreground` | 8 | samples with smaller \|F\| (or \|F\| ≤ n) are masked out |

### Method B (`configs/method_b.yaml`)

| Key | Default | Description |
|---|---|---|
| `method.density.counter_ckpt` | `runs/counter/counter_best.pt` | frozen counter checkpoint |
| `method.density.alpha` | 1.0 | **OPEN.** Under-count weight α |
| `method.density.beta` | 1.0 | **OPEN.** Over-count weight β (includes stray categories) |
| `method.density.variant` | `hinge` | `hinge` (Eq. 16), `tolerance`, `huber` |
| `method.density.tolerance` | 0.25 | δ_c for `tolerance` |
| `method.density.huber_delta` | 1.0 | Huber δ for `huber` |
| `method.density.timestep_weight` | `constant` | w(t): `constant`, `alpha_bar`, `threshold` |
| `method.density.timestep_threshold` | 500 | τ for `threshold` |

### Method C (`configs/method_c.yaml`)

| Key | Default | Description |
|---|---|---|
| `method.attention.cross_layers` | `[up_blocks.2.attentions.0.transformer_blocks.0.attn2]` | **OPEN** (layer ℓ). Layer(s) for ā |
| `method.attention.smoothing_sigma` | 1.0 | smoothing of ā, in tokens |
| `method.attention.grid` | null | resample ā to this `[h, w]` grid (default: the first layer's grid) |
| `method.peak.kmeans_iters` | 3 | **OPEN.** J soft k-means iterations |
| `method.peak.tau_q` | 4.0 | **OPEN.** Assignment temperature (tokens²) |
| `method.peak.tau_p` | 0.1 | **OPEN.** Soft-max temperature for a_k |
| `method.peak.r_min` | 3.0 | **OPEN.** Repulsion radius (tokens). Set from shape size |
| `method.peak.r_s` | 2.5 | **OPEN.** Suppression radius (tokens) |
| `method.peak.peak_min_value` | 0.1 | minimum ā for maxima that seed centres |
| `method.peak.supp_min_value` | 0.0 | minimum ā for maxima counted in L_supp |
| `method.peak.fps_min_value` | 0.0 | farthest-point candidates need ā ≥ this |
| `method.peak.eps` | 1e-6 | ε in Eq. 19 |
| `method.peak.diag_peak_threshold` | 0.3 | threshold of the `C/*` peak diagnostics |

## eval (§8, `scripts/evaluate.py`)

| Key | Default | Description |
|---|---|---|
| `eval.split` | `heldout` | shapes to evaluate: `heldout`, `train`, `all` |
| `eval.seeds` | 16 | number of seeds (0…seeds−1). The same seed is used for every count, which gives the per-seed histograms |
| `eval.counts` | null | counts to evaluate (null = 1…N_max) |
| `eval.hard_counter` | `color` | `color` (connected components) or `owlv2` |
| `eval.save_images` | true | save image grids per (colour, shape) |
| `eval.clip` | false | compute CLIP score (downloads `openai/clip-vit-base-patch32`) |

## precheck (§8 pre-checks, §9 offline scan)

| Key | Default | Description |
|---|---|---|
| `precheck.sources` | `[synthetic, generated]` | data sources, see [workflow.md](workflow.md) |
| `precheck.counts` | 1…10 | true counts to test |
| `precheck.samples_per_count` | 8 | samples per count |
| `precheck.batch_size` | 8 | U-Net batch |
| `precheck.seed` | 0 | seed for scene and prompt sampling |
| `precheck.timesteps` | `[999, 800, 600, 400, 200]` | noise levels for the synthetic source |
| `precheck.resolutions` | `[16, 32]` | token-grid sides of the layers to scan |
| `precheck.touching_frac` | 0.4 | share of scenes with touching shapes |
| `precheck.identical_frac` | 0.5 | share of scenes with identical shapes |
| `precheck.radius_range` | `[0.045, 0.09]` | shape radius as a fraction of the image |
| `precheck.hard_counter` | `color` | counter giving the true count for the generated source |
| `precheck.A.affinities` | `[attention, feature_cosine]` | affinities to scan |
| `precheck.A.gammas` | `[1, 2, 4]` | γ grid |
| `precheck.A.taus` | `[0.05, 0.1, 0.2]` | τ grid |
| `precheck.A.Ts` | `[0.01, 0.02, 0.05]` | T grid |
| `precheck.A.K` | 256 | foreground cap |
| `precheck.A.tau_a` | 0.3 | ā threshold |
| `precheck.A.smoothing_sigma` | 0.5 | ā smoothing |
| `precheck.A.ridge` | 1e-3 | ρ |
| `precheck.A.ridge_mode` | `relative` | ρ mode |
| `precheck.A.drop_self_loops` | false | zero W_ii |
| `precheck.C.smoothing_sigmas` | `[0.0, 1.0]` | ā smoothing grid |
| `precheck.C.thresholds` | `[0.2, 0.3, 0.5]` | peak-threshold grid |

## counter (C_ψ training, §4.1, `configs/counter.yaml`)

| Key | Default | Description |
|---|---|---|
| `counter.input_space` | `latent` | `latent` (reads x̂0 directly) or `image` (decoded x̂0) |
| `counter.image_size` | 256 | input size in `image` mode |
| `counter.latent_channels` | 4 | input channels in `latent` mode |
| `counter.colors` | null | category colours (null = `data.colors`) |
| `counter.shapes` | null | category shapes (null = train + held-out shapes) |
| `counter.density_sigma_frac` | 0.025 | Gaussian σ as a fraction of the output side |
| `counter.arch.base_channels` | 64 | width |
| `counter.arch.channel_mult` | `[1, 2, 4]` | levels |
| `counter.arch.out_activation` | `softplus` | `softplus` or `relu` |
| `counter.arch.head_bias_init` | -10.0 | initial head bias (strongly negative, §4.1) |
| `counter.dataset.train_size` | 20000 | training scenes |
| `counter.dataset.val_size` | 2000 | validation scenes |
| `counter.dataset.max_categories_per_image` | 3 | categories per scene |
| `counter.dataset.max_total_objects` | 20 | object cap per scene |
| `counter.dataset.empty_prob` | 0.05 | share of empty scenes |
| `counter.dataset.touching_prob` | 0.3 | per-shape touching probability (in half the scenes) |
| `counter.dataset.identical_prob` | 0.3 | share of scenes with identical shapes per category |
| `counter.dataset.heldout_combo_fraction` | 0.1 | held-out (category, count) pairs |
| `counter.dataset.radius_range` | `[0.045, 0.09]` | shape radius as a fraction of the image |
| `counter.dataset.background` | `random_light` | `white`, `random_light` or an RGB list |
| `counter.dataset.cache_dir` | `data_cache/counter` | latent cache location |
| `counter.dataset.harvested` | `[]` | extra latent caches from `scripts/harvest_generator_samples.py` |
| `counter.image_aug` | `{prob, contrast, blur_sigma_max, noise_std}` | augmentation before VAE encoding (latent mode, training split) |
| `counter.online_aug` | `{prob, contrast, blur_sigma_max, noise_std}` | augmentation of every training batch |
| `counter.train.epochs` | 30 | epochs |
| `counter.train.batch_size` | 64 | batch size |
| `counter.train.lr` | 3e-4 | AdamW learning rate (cosine schedule) |
| `counter.train.weight_decay` | 1e-4 | AdamW weight decay |
| `counter.train.mse_scale` | 100.0 | s in MSE(s·D, s·target) |
| `counter.train.l1_weight` | 0.1 | weight of the count L1 term |
| `counter.train.num_workers` | 4 | data-loader workers |
| `counter.train.log_every` | 50 | log interval (steps) |
