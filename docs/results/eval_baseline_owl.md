# eval_baseline_owl - frozen SD-Turbo reference

## Provenance

- Commit: `f9e3559` on `claude/cs485-gpu-setup-5bb57e`
- **Working tree dirty**: 18 modified file(s), so the commit above does not fully describe this run
- Python 3.12.10 on Windows 11
- torch 2.11.0+cu128, CUDA 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8.5 GB, compute capability sm_120

## Evaluation - eval_baseline_owl

Source: `C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/eval_baseline_owl`

**Count accuracy**

| metric | overall | heldout_shapes |
|---|---|---|
| `n` | 2880 | 2880 |
| `exact_acc` | 0.1181 | 0.1181 |
| `within1_acc` | 0.2392 | 0.2392 |
| `mae` | 4.8799 | 4.8799 |
| `rel_err` | 0.9347 | 0.9347 |

**Exact accuracy by requested count**

| requested | overall | heldout_shapes |
|---|---|---|
| n=1 | 0.6424 | 0.6424 |
| n=2 | 0.1910 | 0.1910 |
| n=3 | 0.0833 | 0.0833 |
| n=4 | 0.0451 | 0.0451 |
| n=5 | 0.0486 | 0.0486 |
| n=6 | 0.0174 | 0.0174 |
| n=7 | 0.0312 | 0.0312 |
| n=8 | 0.0625 | 0.0625 |
| n=9 | 0.0312 | 0.0312 |
| n=10 | 0.0278 | 0.0278 |

**Run-level metrics**

| metric | value |
|---|---|
| `latency_ms_per_image` | 2101.0456 |
| `diversity` | 0.4609 |
| `nn_regularity_cv` | 0.2906 |
| `seed_spearman_mean` | 0.3017 |
| `seed_distinct_counts_mean` | 6.0764 |

**Count response.** Slope of counted against requested: **+0.696** overall (1.0 = perfect tracking, 0.0 = prompt ignored).

A single slope hides a saturating curve, so it is also split at n=3, which is where the frozen baseline stops following the prompt:

- n ≤ 3: **+2.488**
- n ≥ 4: **+0.273**

A sampler that is really counting raises the second number. Raising only the first means it sharpened a range the frozen model already handled.

| requested n | counted mean | median | range | exact | bias |
|---|---|---|---|---|---|
| 1 | 1.21 | 1.0 | 0-14 | 0.642 | +0.21 |
| 2 | 3.18 | 2.0 | 0-20 | 0.191 | +1.18 |
| 3 | 6.18 | 6.0 | 0-26 | 0.083 | +3.18 |
| 4 | 6.06 | 6.0 | 0-21 | 0.045 | +2.06 |
| 5 | 7.68 | 8.0 | 0-31 | 0.049 | +2.68 |
| 6 | 8.27 | 9.0 | 0-28 | 0.017 | +2.27 |
| 7 | 8.19 | 8.0 | 0-28 | 0.031 | +1.19 |
| 8 | 7.60 | 7.0 | 0-28 | 0.062 | -0.40 |
| 9 | 8.34 | 8.0 | 0-28 | 0.031 | -0.66 |
| 10 | 8.39 | 7.0 | 0-32 | 0.028 | -1.61 |

**Is the count signal real?** A model that ignores the prompt can still post a respectable MAE, so the reference below is the best constant prediction on this very set. Skill is the fraction of that bar's error removed; 0 means no better than guessing one number for every prompt.

| quantity | value |
|---|---|
| records | 2880 |
| counted mean / median / max | 6.51 / 5.0 / 32 |
| requested mean / median / max | 5.50 / 5.5 / 10 |
| MAE | 4.8799 |
| best-constant MAE (always answer 5) | 2.5000 |
| **skill vs constant** (1 - MAE/bar) | **-0.952** |
| Pearson r(counted, requested) | 0.310 |
