# eval_baseline_fix

## Provenance

- Commit: `9cfc999` on `claude/cs485-gpu-setup-5bb57e`
- **Working tree dirty**: 1 modified file(s), so the commit above does not fully describe this run
- Python 3.12.10 on Windows 11
- torch 2.11.0+cu128, CUDA 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8.5 GB, compute capability sm_120

## Evaluation - eval_baseline_fix

Source: `C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/eval_baseline_fix`

**Count accuracy**

| metric | overall | heldout_shapes |
|---|---|---|
| `n` | 2880 | 2880 |
| `exact_acc` | 0.0622 | 0.0622 |
| `within1_acc` | 0.1760 | 0.1760 |
| `mae` | 8.1660 | 8.1660 |
| `rel_err` | 1.6744 | 1.6744 |

**Exact accuracy by requested count**

| requested | overall | heldout_shapes |
|---|---|---|
| n=1 | 0.3229 | 0.3229 |
| n=2 | 0.0868 | 0.0868 |
| n=3 | 0.0278 | 0.0278 |
| n=4 | 0.0208 | 0.0208 |
| n=5 | 0.0382 | 0.0382 |
| n=6 | 0.0243 | 0.0243 |
| n=7 | 0.0243 | 0.0243 |
| n=8 | 0.0451 | 0.0451 |
| n=9 | 0.0174 | 0.0174 |
| n=10 | 0.0139 | 0.0139 |

**Run-level metrics**

| metric | value |
|---|---|
| `latency_ms_per_image` | 1821.0787 |
| `diversity` | 0.4609 |
| `nn_regularity_cv` | 0.5128 |
| `seed_spearman_mean` | 0.6666 |
| `seed_distinct_counts_mean` | 7.8090 |

**Count response.** Slope of counted against requested: **+1.643** overall (1.0 = perfect tracking, 0.0 = prompt ignored).

A single slope hides a saturating curve, so it is also split at n=3, which is where the frozen baseline stops following the prompt:

- n ≤ 3: **+3.828**
- n ≥ 4: **+1.031**

A sampler that is really counting raises the second number. Raising only the first means it sharpened a range the frozen model already handled.

| requested n | counted mean | median | range | exact | bias |
|---|---|---|---|---|---|
| 1 | 2.57 | 2.0 | 0-32 | 0.323 | +1.57 |
| 2 | 6.08 | 5.0 | 0-37 | 0.087 | +4.08 |
| 3 | 10.23 | 9.0 | 0-31 | 0.028 | +7.23 |
| 4 | 11.64 | 11.0 | 0-40 | 0.021 | +7.64 |
| 5 | 14.34 | 14.0 | 0-44 | 0.038 | +9.34 |
| 6 | 15.98 | 15.0 | 0-48 | 0.024 | +9.98 |
| 7 | 15.74 | 16.0 | 0-47 | 0.024 | +8.74 |
| 8 | 15.38 | 15.0 | 0-43 | 0.045 | +7.38 |
| 9 | 17.85 | 18.0 | 1-44 | 0.017 | +8.85 |
| 10 | 19.12 | 19.0 | 1-38 | 0.014 | +9.12 |

**Is the count signal real?** A model that ignores the prompt can still post a respectable MAE, so the reference below is the best constant prediction on this very set. Skill is the fraction of that bar's error removed; 0 means no better than guessing one number for every prompt.

| quantity | value |
|---|---|
| records | 2880 |
| counted mean / median / max | 12.89 / 12.0 / 48 |
| requested mean / median / max | 5.50 / 5.5 / 10 |
| MAE | 8.1660 |
| best-constant MAE (always answer 5) | 2.5000 |
| **skill vs constant** (1 - MAE/bar) | **-2.266** |
| Pearson r(counted, requested) | 0.529 |
