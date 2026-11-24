# c_tuned_s0

## Provenance

- Commit: `9cfc999` on `claude/cs485-gpu-setup-5bb57e`
- Python 3.12.10 on Windows 11
- torch 2.11.0+cu128, CUDA 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8.5 GB, compute capability sm_120

## Resolved configuration

`C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/c_tuned_s0/config.yaml`

```yaml
experiment:
  name: c_tuned_s0
  output_dir: C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs
  seed: 0
  device: auto
  wandb: false
  wandb_project: count-sampler
generator:
  model_id: stabilityai/sd-turbo
  family: auto
  variant: fp16
  dtype: bfloat16
  resolution: 512
  timesteps:
  - 999
  input_scaling: pipeline
  gradient_checkpointing: true
  min_alpha_bar: 0.0001
data:
  n_max: 10
  colors:
  - red
  - green
  - blue
  - yellow
  - orange
  - purple
  train_shapes:
  - circle
  - square
  - triangle
  - star
  - hexagon
  heldout_shapes:
  - pentagon
  - diamond
  - cross
  templates:
  - '{count} {color} {shape}'
  - a photo of {count} {color} {shape}
  - '{count} {color} {shape} on a white background'
  count_format: word
  include_count: true
sampler:
  w: 1.0
  base_channels: 64
  channel_mult:
  - 1
  - 2
  - 2
  blocks_per_level: 1
  mid_attention: true
  cond_dim: 256
  count_embed_dim: 64
  count_embed_max_freq: 64.0
  text_embed_dim: 128
  use_text: true
geometry:
  delta: 0.3
  g: log
  ord_form: regression
  rank_margin: 0.05
  over: sampled
train:
  steps: 6000
  eps_per_batch: 1
  counts_per_eps: 4
  grad_accum: 2
  lr: 0.0001
  weight_decay: 0.0
  warmup_steps: 100
  grad_clip: 1.0
  log_every: 10
  ckpt_every: 1000
  resume: null
  preview:
    every: 500
    seeds:
    - 0
    - 1
    counts:
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 8
    - 10
    prompts:
    - - red
      - circle
    - - blue
      - pentagon
eval:
  split: heldout
  seeds: 16
  counts: null
  hard_counter: color
  save_images: true
  clip: false
  benchmark:
    name: null
    path: null
    seeds: 4
    dedupe: false
    counter: owlv2
    counter_kwargs: {}
    multi_object_count: sum
    batch_size: 8
    save_images: false
precheck:
  sources:
  - synthetic
  - generated
  counts:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
  - 10
  samples_per_count: 8
  batch_size: 4
  seed: 0
  timesteps:
  - 999
  - 800
  - 600
  - 400
  - 200
  resolutions:
  - 16
  - 32
  touching_frac: 0.4
  identical_frac: 0.5
  radius_range:
  - 0.045
  - 0.09
  hard_counter: color
  A:
    affinities:
    - attention
    - feature_cosine
    gammas:
    - 1.0
    - 2.0
    - 4.0
    taus:
    - 0.05
    - 0.1
    - 0.2
    Ts:
    - 0.01
    - 0.02
    - 0.05
    K: 256
    tau_a: 0.3
    smoothing_sigma: 0.5
    ridge: 0.001
    ridge_mode: relative
    drop_self_loops: false
  C:
    smoothing_sigmas:
    - 0.0
    - 1.0
    thresholds:
    - 0.2
    - 0.3
    - 0.5
method:
  name: C
  weights:
    act: 1.0
    rep: 1.0
    supp: 1.0
    anc: 10.0
    pri: 10.0
    ord: 1.0
  stages:
  - start: 0
    terms:
    - act
    - supp
    - anc
    - pri
  - start: 2000
    terms:
    - act
    - supp
    - rep
    - anc
    - pri
  - start: 4000
    terms:
    - act
    - supp
    - rep
    - anc
    - pri
    - ord
  loss_steps: all
  attention:
    cross_layers:
    - up_blocks.1.attentions.1.transformer_blocks.0.attn2
    smoothing_sigma: 1.0
    grid: null
  peak:
    kmeans_iters: 3
    tau_q: 4.0
    tau_p: 0.1
    r_min: 1.5
    r_s: 1.25
    peak_min_value: 0.1
    supp_min_value: 0.0
    fps_min_value: 0.0
    eps: 1.0e-06
    diag_peak_threshold: 0.5
counter:
  train:
    batch_size: 32
```

## Checkpoints

| file | step | parameters | size | sha256[:16] |
|---|---|---|---|---|
| `sampler_step001000.pt` | 1000 | 3,202,628 | 38.6 MB | `c68ec4932ddf9805` |
| `sampler_step002000.pt` | 2000 | 3,202,628 | 38.6 MB | `ad557383b7ada759` |
| `sampler_step003000.pt` | 3000 | 3,202,628 | 38.6 MB | `3f3c190e6edd9f0b` |
| `sampler_step004000.pt` | 4000 | 3,202,628 | 38.6 MB | `df150a7a674f39ee` |
| `sampler_step005000.pt` | 5000 | 3,202,628 | 38.6 MB | `c291625acde42d9f` |
| `sampler_step006000.pt` | 6000 | 3,202,628 | 38.6 MB | `f8f3944f4c714c51` |
| `sampler_last.pt` | 6000 | 3,202,628 | 38.6 MB | `15c8b336a77ef51a` |

## Training

Logged lines: **601**, steps 1 to 6000, wall time **124.2 min**.
- `sec_per_step`: mean 1.003, median 0.840
- peak `gpu_mem_gb`: 6.73
- skipped optimiser steps: **0** of 601 logged; dropped micro-batches: **0**

**Mean of each logged scalar, by stage**

| key | stage 1.0 (n=201) | stage 2.0 (n=200) | stage 3.0 (n=200) |
|---|---|---|---|
| `C/frac_peaks_ge_n` | 0.3004 | 0.2175 | 0.1613 |
| `C/num_peaks_minus_n` | -2.643 | -3.388 | -3.506 |
| `C/peak_value_mean` | 0.8464 | 0.9268 | 0.9322 |
| `geom/d_anchor_max` | 0.268 | 0.2946 | 0.302 |
| `geom/d_anchor_mean` | 0.2615 | 0.2925 | 0.294 |
| `geom/d_pair_mean` | 0.06019 | 0.06481 | 0.2138 |
| `geom/norm_sq` | 1.006 | 1 | 1.001 |
| `grad_norm` | 19.97 | 2.179 | 2.828 |
| `loss/act` | 0.1796 | 0.07778 | 0.07204 |
| `loss/anc` | 3.914e-05 | 2.284e-05 | 6.666e-05 |
| `loss/ord` | 0.289 | 0.2281 | 0.006012 |
| `loss/pri` | 0.001217 | 0.0004902 | 0.0006042 |
| `loss/rep` | 1.735e-06 | 0 | 9.024e-08 |
| `loss/supp` | 0.883 | 0.3433 | 0.2755 |
| `loss/total` | 1.075 | 0.4262 | 0.3603 |
| `lr` | 9.732e-05 | 0.0001 | 0.0001 |

**First vs last 60 logged steps** (did the optimised quantity move?)

| key | first 60 | last 60 | change |
|---|---|---|---|
| `C/frac_peaks_ge_n` | 0.3896 | 0.1729 | -0.2167 |
| `C/num_peaks_minus_n` | -1.521 | -3.435 | -1.915 |
| `C/peak_value_mean` | 0.778 | 0.9344 | +0.1564 |
| `geom/d_anchor_max` | 0.2143 | 0.3011 | +0.0868 |
| `geom/d_anchor_mean` | 0.2089 | 0.2916 | +0.08274 |
| `geom/d_pair_mean` | 0.03417 | 0.2178 | +0.1837 |
| `geom/norm_sq` | 1.014 | 1.001 | -0.01274 |
| `grad_norm` | 34.11 | 1.899 | -32.21 |
| `loss/act` | 0.2739 | 0.06836 | -0.2055 |
| `loss/anc` | 1.298e-05 | 3.847e-05 | +2.549e-05 |
| `loss/ord` | 0.369 | 0.003852 | -0.3651 |
| `loss/pri` | 0.002021 | 0.0005509 | -0.00147 |
| `loss/rep` | 5.812e-06 | 0 | -5.812e-06 |
| `loss/supp` | 1.574 | 0.2511 | -1.323 |
| `loss/total` | 1.868 | 0.3292 | -1.539 |
| `lr` | 9.102e-05 | 0.0001 | +8.983e-06 |

## Evaluation - eval_fix

Source: `C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/c_tuned_s0/eval_fix`

**Count accuracy**

| metric | overall | heldout_shapes |
|---|---|---|
| `n` | 2880 | 2880 |
| `exact_acc` | 0.0993 | 0.0993 |
| `within1_acc` | 0.1236 | 0.1236 |
| `mae` | 5.3115 | 5.3115 |
| `rel_err` | 0.8814 | 0.8814 |

**Exact accuracy by requested count**

| requested | overall | heldout_shapes |
|---|---|---|
| n=1 | 0.9618 | 0.9618 |
| n=2 | 0.0312 | 0.0312 |
| n=3 | 0.0000 | 0.0000 |
| n=4 | 0.0000 | 0.0000 |
| n=5 | 0.0000 | 0.0000 |
| n=6 | 0.0000 | 0.0000 |
| n=7 | 0.0000 | 0.0000 |
| n=8 | 0.0000 | 0.0000 |
| n=9 | 0.0000 | 0.0000 |
| n=10 | 0.0000 | 0.0000 |

**Run-level metrics**

| metric | value |
|---|---|
| `latency_ms_per_image` | 2549.9208 |
| `diversity` | 0.1878 |
| `nn_regularity_cv` | 0.7312 |
| `seed_spearman_mean` | -0.4208 |
| `seed_distinct_counts_mean` | 2.0174 |
| `template` | {count} {color} {shape} |

**Count response.** Slope of counted against requested: **-0.045** overall (1.0 = perfect tracking, 0.0 = prompt ignored).

A single slope hides a saturating curve, so it is also split at n=3, which is where the frozen baseline stops following the prompt:

- n ≤ 3: **-0.469**
- n ≥ 4: **+0.023**

A sampler that is really counting raises the second number. Raising only the first means it sharpened a range the frozen model already handled.

| requested n | counted mean | median | range | exact | bias |
|---|---|---|---|---|---|
| 1 | 0.96 | 1.0 | 0-1 | 0.962 | -0.04 |
| 2 | 0.27 | 0.0 | 0-3 | 0.031 | -1.73 |
| 3 | 0.02 | 0.0 | 0-1 | 0.000 | -2.98 |
| 4 | 0.02 | 0.0 | 0-2 | 0.000 | -3.98 |
| 5 | 0.02 | 0.0 | 0-2 | 0.000 | -4.98 |
| 6 | 0.07 | 0.0 | 0-2 | 0.000 | -5.93 |
| 7 | 0.11 | 0.0 | 0-1 | 0.000 | -6.89 |
| 8 | 0.13 | 0.0 | 0-1 | 0.000 | -7.87 |
| 9 | 0.13 | 0.0 | 0-3 | 0.000 | -8.87 |
| 10 | 0.15 | 0.0 | 0-2 | 0.000 | -9.85 |

**Is the count signal real?** A model that ignores the prompt can still post a respectable MAE, so the reference below is the best constant prediction on this very set. Skill is the fraction of that bar's error removed; 0 means no better than guessing one number for every prompt.

| quantity | value |
|---|---|
| records | 2880 |
| counted mean / median / max | 0.19 / 0.0 / 3 |
| requested mean / median / max | 5.50 / 5.5 / 10 |
| MAE | 5.3115 |
| best-constant MAE (always answer 5) | 2.5000 |
| **skill vs constant** (1 - MAE/bar) | **-1.125** |
| Pearson r(counted, requested) | -0.318 |
