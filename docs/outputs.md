# Outputs and logs

`runs/` and `data_cache/` are gitignored. Share checkpoints out of band, and record results in [experiment_log.md](experiment_log.md).

## Directory layout

```
runs/
├── <experiment.name>/              # scripts/train_sampler.py
│   ├── config.yaml                 # fully resolved config
│   ├── log.jsonl                   # one JSON object per logged step
│   ├── previews/step_<N>_<color>_<shape>.png
│   ├── sampler_step<N>.pt          # every train.ckpt_every steps
│   ├── sampler_last.pt
│   └── eval/                       # scripts/evaluate.py --ckpt (default location)
│       ├── metrics.json
│       ├── records.csv
│       ├── per_seed_hist.csv
│       └── grids/<color>_<shape>.png
├── counter/                        # scripts/train_counter.py
│   ├── config.yaml, log.jsonl
│   ├── counter_best.pt             # lowest val/mae_present
│   └── counter_last.pt
├── eval_baseline/                  # scripts/evaluate.py --baseline (same files as eval/)
└── precheck/                       # scripts/precheck.py
    ├── precheck_A_rows.csv, precheck_A_summary.csv
    ├── precheck_B_rows.csv, precheck_B_summary.csv
    └── precheck_C_rows.csv, precheck_C_summary.csv
data_cache/counter/{train,val}_<config-hash>.pt   # VAE-encoded synthetic scenes
```

## Sampler training log (`log.jsonl`)

Each line has `step` and `elapsed_s` plus the keys below. With `train.grad_accum > 1`, values are averaged over micro-batches.
With multi-step unrolls, method diagnostics get a `@t<timestep>` suffix.

| Key | Meaning |
|---|---|
| `loss/total` | weighted sum of the **active** terms (the quantity that is optimised) |
| `loss/<term>` | every term, active or not: `cnt`, `gap` (A), `den` (B), `act`, `rep`, `supp` (C), and `anc`, `pri`, `ord` |
| `stage` | current stage number (1-based) of `method.stages` |
| `geom/d_anchor_mean`, `geom/d_anchor_max` | d(z_n, ε). Should stay ≤ `geometry.delta` |
| `geom/norm_sq` | mean ‖z_n‖²/D. Should stay ≈ 1 |
| `geom/d_pair_mean` | mean d(z_n, z_m) between counts of the same ε |
| `A/k_hat_mean`, `A/k_hat_abs_err`, `A/lambda_next_mean`, `A/n_fg_mean`, `A/valid_frac`, `A/gamma`, `A/T` | Method A, see [method_a_spectral.md](methods/method_a_spectral.md#diagnostics-log-keys) |
| `B/count_abs_err`, `B/count_exact`, `B/stray_mass` | Method B, see [method_b_density.md](methods/method_b_density.md#diagnostics-log-keys) |
| `C/frac_peaks_ge_n`, `C/num_peaks_minus_n`, `C/peak_value_mean` | Method C, see [method_c_peak.md](methods/method_c_peak.md#diagnostics-log-keys) |
| `skipped` | 1 if the optimiser step was skipped (every micro-batch failed, or the gradient norm was non-finite), else 0 |
| `skipped_micro` | number of micro-batches dropped (non-finite loss or failed eigendecomposition) |
| `grad_norm` | gradient norm before clipping |
| `lr` | learning rate |
| `sec_per_step` | wall time of the step |
| `gpu_mem_gb` | peak GPU memory so far (CUDA only) |

Read it in Python:

```python
import json
rows = [json.loads(line) for line in open("runs/method_a/log.jsonl")]
steps = [r["step"] for r in rows]
err = [r.get("A/k_hat_abs_err") for r in rows]
```

## Preview grids

`previews/step_<N>_<color>_<shape>.png`: for each seed in `train.preview.seeds`, two rows (the **baseline** z = ε, then the
**sampler** z_n), with one column per count in `train.preview.counts`. Step 0 shows identical rows because the sampler starts as the identity.

## Counter training log

Training lines: `epoch`, `loss`, `mse`, `l1`, `lr`. One validation line per epoch:

| Key | Meaning |
|---|---|
| `val/mae_all`, `val/exact_all` | over every (image, category) pair, absent categories included |
| `val/mae_present`, `val/exact_present` | over categories present in the image (used to pick `counter_best.pt`) |
| `val/stray_mass` | mean predicted count in absent categories |
| `val/mae_heldout_combo`, `val/exact_heldout_combo` | (category, count) pairs never seen in training |

## Evaluation (`scripts/evaluate.py`)

**`metrics.json`**

| Key | Meaning |
|---|---|
| `overall`, `heldout_shapes`, `train_shapes` | count metrics on all records / held-out shapes / training shapes: `n`, `exact_acc`, `within1_acc`, `mae`, `rel_err`, `exact_acc@<n>` |
| `latency_ms_per_image` | sampler + generator + VAE decode, per image |
| `diversity` | mean pairwise distance between images of the same prompt across seeds |
| `nn_regularity_cv` | coefficient of variation of nearest-neighbour distances between detected objects. Low means grid-like |
| `clip_score` | mean CLIP prompt–image similarity (only with `--clip`) |
| `seed_spearman_mean` | per seed, Spearman correlation between requested n and counted value, averaged. Higher means the count follows the prompt |
| `seed_distinct_counts_mean` | per seed, number of distinct counted values across n |

**`records.csv`**: one row per image: `color, shape, heldout_shape, seed, n, counted, regularity, clip`.

**`per_seed_hist.csv`**: one row per (seed, colour, shape), with the counted value for each requested n plus
`spearman` and `distinct_counts`. This is the "per-seed count histogram" of §8. Compare baseline and trained runs row by row.

## Pre-checks (`scripts/precheck.py`)

| File | Columns |
|---|---|
| `precheck_A_rows.csv` | `source, t, layer, affinity, gamma, tau, T, n_prompt, n_true, touching, identical, k_hat, n_fg` |
| `precheck_A_summary.csv` | grouped by `source, t, layer, affinity, gamma, tau, T`: `n, mae, exact, pearson, mean_pred_minus_true`, sorted by `mae` |
| `precheck_B_rows.csv` | `source, t, n_prompt, n_true, n_hat, stray, touching` |
| `precheck_B_summary.csv` | grouped by `source, t`: `n, mae, exact, pearson, mean_pred_minus_true` |
| `precheck_C_rows.csv` | `source, t, layer, sigma, threshold, n_prompt, n_true, touching, num_peaks` |
| `precheck_C_summary.csv` | grouped by `source, t, layer, sigma, threshold`: `n, frac_peaks_ge_n, frac_peaks_eq_n, mae`, sorted by `frac_peaks_ge_n` |

`n_true` is the rendered count (synthetic source) or the hard counter's count of the decoded image (generated source).
Rows with `n_true = 0` are left out of the summaries.

## Checkpoint formats

Both formats are plain dicts of tensors and Python primitives, so they load with `torch.load(..., weights_only=True)`.

**Sampler** (`sampler_*.pt`): `state_dict`, `sampler_config`, `latent_channels`, `text_dim`, `step`, `config` (the full
resolved config), `optimizer`, `lr_scheduler`.

```python
from countsampler.training import load_sampler_checkpoint
sampler, ckpt = load_sampler_checkpoint("runs/method_a/sampler_last.pt", device="cuda")
z = sampler(eps, n, pooled_c)          # eps [B,4,64,64], n [B] long, pooled_c [B, text_dim]
```

**Counter** (`counter_*.pt`): `state_dict`, `meta` = {`in_channels`, `categories`, `input_space`, `image_size`, `arch`,
`density_sigma`, `heldout_combos`, `model_id`, `metrics`, `epoch`}.

```python
from countsampler.novel.components import load_counter
counter, meta = load_counter("runs/counter/counter_best.pt", device="cuda")   # frozen
```
