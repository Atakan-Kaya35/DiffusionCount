# Count-Conditioned Noise Sampler (CS485, Bilkent)

Implementation of **"Count-Conditioned Noise Sampler: Three Methodologies" (draft v5)**,
[`count_sampler_formalization_v5.pdf`](count_sampler_formalization_v5.pdf).

A small network Δφ changes the initial noise of a **frozen** few-step text-to-image model
(SD-Turbo / SDXL-Turbo) so that the image contains the requested number of objects:

```
z_n = ε + w · Δφ(ε, n, c̄)          (Eq. 2. Only φ is trained.)
```

The sampler is the same in all experiments. Only the error signal that trains it changes:

| Method | Error signal | Extra module | Code |
|---|---|---|---|
| **A** spectral | eigengap soft count on the self-attention graph of the cross-attention foreground | none | `novel/losses/spectral.py` |
| **B** density | frozen density-map counter C_ψ on the one-step clean estimate x̂0 | counter C_ψ | `novel/losses/density.py`, `novel/components/counter.py` |
| **C** peak | separated cross-attention peaks of the object token (soft centres) | none | `novel/losses/peak.py` |

All three share the sampler and its geometry regularizers (L_anc, L_pri, optionally L_ord) in `novel/shared.py`.

---

## Repository layout

```
countsampler/
├── novel/                  ★ RESEARCH CODE: everything we tune or change to get results
│   ├── shared.py           §2   sampler Δφ + geometry losses (one file on purpose)
│   ├── losses/             one file per method
│   │   ├── spectral.py     §3   Method A
│   │   ├── density.py      §4   Method B
│   │   └── peak.py         §5   Method C
│   ├── components/
│   │   └── counter.py      §4.1 density counter C_ψ
│   └── methods/            loss weighting + stage schedules (Eq. 15, 17, 22)
│       ├── base.py
│       ├── method_a.py / method_b.py / method_c.py
├── generator/              frozen SD-Turbo / SDXL-Turbo wrapper, attention capture, Eq. (1)
├── data/                   prompts p_n, synthetic shapes, counter datasets, generator harvesting
├── training/               sampler loop (§7), counter training (§4.1), checkpoints
├── evaluation/             hard counters, metrics, evaluation (§8), pre-checks / offline scan (§8-9)
└── utils/                  config, logging, misc
configs/                    YAML. Every spec knob is here, and open choices (§9) are marked OPEN
scripts/                    CLI entry points
tests/                      CPU tests on tiny random models (no downloads)
```

Every file in `countsampler/novel/` starts with a **★ NOVEL ★** banner listing the equations it
implements and where it comes from (§10). Code outside `novel/` is plumbing and should rarely need changes.

## Spec → code map

| Spec | What | Where |
|---|---|---|
| Eq. 1 | clean estimate x̂0 | `generator/turbo.py` `FrozenTurboGenerator.forward_at` |
| Eq. 2 | sampler z_n = ε + wΔφ(ε, n, c̄), zero-init last layer, FiLM on sin-enc(log n) and c̄ | `novel/shared.py` `CountNoiseSampler` |
| Eq. 3-4, 8 | d(a,b), g_log / g_lin, α = 2δ/g_max | `novel/shared.py` |
| Eq. 5 / 6 | L_ord regression / ranking | `GeometryLosses.ordering_loss` / `ranking_loss` |
| Eq. 7, 9 | L_anc, L_pri | `GeometryLosses.anchor_loss`, `prior_loss` |
| §3.1 | ā (head-avg, smoothed, min-max), explicit S = softmax(QKᵀ/√d), foreground F under stop-grad | `generator/attention.py`, `spectral.select_foreground` |
| Eq. 10-11 | affinity W = (½(S+Sᵀ))^γ, normalized Laplacian with ridge ρ | `spectral.attention_affinity`, `normalized_laplacian` |
| Eq. 12-14 | soft count k̂, L_cnt (Huber on log), L_gap | `spectral.soft_count`, `count_loss`, `gap_loss` |
| §3.3 | float64 batched `eigvalsh`, γ/T annealing, grad clipping | `spectral.py`, `methods/method_a.py`, `training/sampler_trainer.py` |
| Remark 2 | cosine-of-self-attention-features fallback | `spectral.feature_affinity` (`method.spectral.affinity: feature_cosine`) |
| Eq. 15 | L_A + stage schedule | `methods/method_a.py`, `configs/method_a.yaml` |
| §4.1 | counter C_ψ, softplus + negative head bias, Gaussian targets, MSE(s·) + L1, augmentation, frozen without no_grad | `components/counter.py`, `training/counter_trainer.py`, `data/counter_data.py` |
| Eq. 16 | L_den, n_k = 0 for unmentioned categories, tolerance / Huber variants, w(t) | `novel/losses/density.py` |
| Eq. 17 | L_B + stage schedule | `methods/method_b.py`, `configs/method_b.yaml` |
| Eq. 18 | soft centres: local-max init (stop-grad) + farthest-point fill + soft k-means | `peak.init_centers`, `soft_kmeans` |
| Eq. 19-21 | L_act, L_rep, L_supp | `peak.peak_losses` |
| Eq. 22 | L_C + stage schedule | `methods/method_c.py`, `configs/method_c.yaml` |
| §7 | training procedure (J counts per ε incl. label count, skip-on-NaN, clipping) and ablation without the number | `training/sampler_trainer.py`, `data.include_count=false` |
| §8 | hard-counter evaluation, per-seed histograms, latency, diversity, regular-grid check | `evaluation/evaluate.py`, `scripts/evaluate.py`, `scripts/compare.py` |
| §8 pre-checks / §9 offline scan | A: k̂ vs count across layers/timesteps/γ/τ/T, both affinities. B: counter on x̂0. C: ≥ n peaks | `evaluation/prechecks.py`, `scripts/precheck.py` |

---

## Setup (on the GPU machine)

RTX 50-series cards (5060 / 5070 Ti, Blackwell `sm_120`) need a PyTorch build with **CUDA 12.8+**:

```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -e ".[dev]"
```

The first run downloads `stabilityai/sd-turbo` (about 2.6 GB with fp16 weight files) from the Hugging Face Hub.
SDXL-Turbo requires accepting its licence on the Hub (`huggingface-cli login`).

## Workflow

**0. Inspect the attention layers** (names and token grids, for `method.attention.*`):
```bash
python scripts/list_layers.py --config configs/base.yaml
```

**1. Pre-checks and offline scan (§8, §9).** Run these before training. They set ℓ, t, γ, τ, T and check that each method's assumption holds on the frozen model:
```bash
python scripts/precheck.py --config configs/base.yaml --methods A C --out-dir runs/precheck
```
Outputs `runs/precheck/precheck_{A,C}_summary.csv`, sorted by MAE, plus the per-sample rows.

**2. Method B only: train the counter once (§4.1).**
```bash
python scripts/train_counter.py --config configs/counter.yaml
```
```bash
python scripts/precheck.py --config configs/base.yaml --methods B --counter-ckpt runs/counter/counter_best.pt
```
Optional, to close the gap between synthetic images and the blurrier x̂0 (§4.1): harvest labelled generator samples, then retrain:
```bash
python scripts/harvest_generator_samples.py --config configs/counter.yaml --out data_cache/counter/harvest.pt
```
```bash
python scripts/train_counter.py --config configs/counter.yaml --set "counter.dataset.harvested=[data_cache/counter/harvest.pt]"
```

**3. Train the sampler**, one method at a time (§6: methods run separately first):
```bash
python scripts/train_sampler.py --config configs/method_a.yaml
```
```bash
python scripts/train_sampler.py --config configs/method_b.yaml
```
```bash
python scripts/train_sampler.py --config configs/method_c.yaml
```
Logs go to `runs/<name>/log.jsonl`. Preview grids (rows: seed × {baseline, sampler}, columns: counts) go to `runs/<name>/previews/`.

Variants are config overrides, so no code changes are needed:
```bash
python scripts/train_sampler.py --config configs/method_a.yaml --set geometry.g=lin experiment.name=a_glin
```
```bash
python scripts/train_sampler.py --config configs/method_a.yaml --set data.include_count=false experiment.name=a_nonum
```

**4. Evaluate and compare (§8).**
```bash
python scripts/evaluate.py --baseline --config configs/base.yaml --out-dir runs/eval_baseline
```
```bash
python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt
```
```bash
python scripts/compare.py runs/eval_baseline runs/method_a/eval runs/method_b/eval runs/method_c/eval
```
The default hard counter (`color`) counts connected components of the prompted colour. It is independent of C_ψ.
`--counter owlv2` uses an open-vocabulary detector instead. CountGD is not pip-installable: wrap it in a class
with the same `count(image, color, shape)` method in `evaluation/hard_counters.py`. D2D is an external baseline.

### GPU memory (RTX 5060 / 5070 Ti)

Defaults target 16 GB: SD-Turbo in bf16, gradient checkpointing, U-Net batch = `eps_per_batch × counts_per_eps` = 2 × 4.
These defaults have **not yet been profiled on the target cards.** Check `gpu_mem_gb` in the log on the first run.
- 8 GB cards: add `--config configs/gpu_8gb.yaml` (halves the micro-batch, keeps the effective batch through `grad_accum`).
- SDXL-Turbo: add `--config configs/sdxl_turbo.yaml` (2.6B U-Net, sets the matching layer names and a smaller micro-batch).
- Method A: keep the self-attention layer at mid resolution (16×16 or 32×32). 64×64 makes S 4096 × 4096.
- Method B with `counter.input_space=image` backpropagates through the VAE decoder, which is much heavier than the latent counter.

## Modifying a method

- **Change a loss:** edit the function in `countsampler/novel/losses/<method>.py`. Each equation is its own function.
- **Change weights or stages:** `method.weights` and `method.stages` in `configs/method_*.yaml`.
- **Change the sampler architecture:** `CountNoiseSampler` in `novel/shared.py` (and the `sampler:` config block).
- **Add a method (e.g. a combination, after the separate runs):** subclass `CountMethod` in `novel/methods/`,
  implement `method_terms` (and `attention_request` if it reads attention), register it in `novel/methods/__init__.py`.

## Interpretation choices (where the spec is open or ambiguous)

Each is a config switch unless noted:

| Topic | Choice | Switch |
|---|---|---|
| First U-Net input | `x_T = √(1−ᾱ_T)·z_n`. This reproduces diffusers' SD/SDXL-Turbo pipelines exactly (tested for 1, 2 and 4 steps). The literal "x_t = z_n" differs by a 0.23% input scale | `generator.input_scaling: pipeline \| identity` |
| Multi-step unroll | deterministic re-noising, which equals SD-Turbo's Euler scheduler. SDXL-Turbo's ancestral noise is not added | — |
| Geometry pairs | across the J sampled noises (§7 step 5). Eq. 5's full 1..N_max sum is available | `geometry.over: sampled \| all_counts` |
| Degree ridge ρ | relative to the mean foreground degree, because raw attention probabilities are tiny | `method.spectral.ridge_mode` |
| Self-loops in W | kept, as written | `method.spectral.drop_self_loops` |
| Too-small foreground | samples with \|F\| ≤ n or below `min_foreground` are masked out of L_cnt / L_gap | `method.spectral.min_foreground` |
| "Self-attention features" (fallback) | the self-attention output before `to_out`, per layer L2-normalised | — |
| Counter MSE "scaled by s" | MSE(s·D, s·target), so soft counts still sum to n | `counter.train.mse_scale` |
| Farthest-point fill (Method C) | over all tokens with ā ≥ `fps_min_value` | `method.peak.fps_min_value` |
| Pooled c̄ | CLIP pooled output (SD-Turbo) / text_encoder_2 projection (SDXL) | `sampler.use_text` |

## Tests

```bash
pytest -q
```

39 CPU tests on tiny random U-Net/VAE/CLIP models. They cover: each equation's behaviour (block-diagonal
attention → k̂ = number of groups; collinear embedding satisfies L_ord, as in Remark 1; soft centres converge on peaks),
gradients reaching z through captured attention under gradient checkpointing, equivalence with diffusers'
Euler pipeline, and end-to-end training steps for A/B/C, counter training, pre-checks and evaluation.

## Provenance

See §10 of the spec. In short: the noise-modifier + norm prior follows D2D/ReNO. Self-/cross-attention reading follows
Make It Count. Cross-attention blobs follow CountCluster. The density loss and the peak losses come from teammates'
specs (the peak losses repaired with soft centres here). The sampler form, geometry losses, spectral losses
and stabilisation are proposed here and **untested**.
