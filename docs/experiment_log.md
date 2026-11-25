# Experiment log

One row per run, failed runs included. Keep the newest at the top. For the metrics, paste the relevant columns of
`scripts/compare.py` output. Put longer notes and figures in a subsection below the table.

## Summary of findings so far (2026-11-24)

The sections below are chronological, including the readings that were later corrected — the
corrections are part of the record. This is where they stand now.

**On the frozen generator**
1. **Use `runs/eval_baseline_nms` — see "THE BASELINE, corrected" below.** Every earlier baseline
   figure in this log was measured with a counter that had three separate defects, and both the
   "it saturates" and the "it tracks almost perfectly" readings were artefacts. Corrected: MAE
   **5.210**, skill **−1.084**, exact **0.1455**, spread **6.01**, and for n ≥ 4
   `counted = 0.595·n + 4.96`. Per shape the slope is +1.91 (cross), +0.52 (diamond) and
   **−0.65** (pentagon), so the aggregate describes none of them.
2. The target: **halve the conditional spread** (6.01 → ~3.0), which is where centring plus
   variance reduction first clears the constant bar. Centring alone scores −0.996. Improving MAE
   without moving the spread or the slope means finding the best constant, not counting — Method C
   did exactly that and the degeneracy guard in `scripts/compare_paired.py` now catches it.

**On Method C (the only method measured end to end so far)**
3. It reached **every one of its own objectives** — `loss/total` 3.872 → 0.125, `loss/rep` exactly
   0.0 throughout, `loss/ord` 0.228 → 0.022 — while producing **one object for every requested
   count**. The failure is in the objective, not the optimisation: `skipped` = 0, `norm_sq` ≈ 1.00,
   gradients falling smoothly.
4. Each term constrains a geometric *proxy* — attention-map shape, centre distances, latent
   distances — and every proxy is invariant under merging two objects. Nothing in the set requires
   the image to contain n objects, or to be the right colour: "blue pentagons" render **red** by
   step 4000.
5. Correction: collapse is a **local** minimum behind a barrier (k=2 costs 2.35 against 1.18 for
   one blob and 0.17 for the correct layout), not the global optimum as first recorded.

**On the measurement — the largest finding**
6. The evaluation was measuring the counter more than the generator. The detector query hit proper
   nouns: **"blue cross" returned nothing on 90 % of images, "red cross" on 70 %**, against 2 % for
   "yellow cross", on images full of well-separated crosses. At n ≥ 5 exact accuracy was **3.65 %
   against 3.72 % chance**.
7. Repaired (threshold 0.3 → 0.1, colour dropped from the query). On rendered scenes with counts
   known by construction the repaired counter scores **MAE 0.143** and the colour counter
   **0.000** — but the two agree at only **r = 0.131** on generated images. A counter that is
   perfect on real scenes and useless on generated ones means **the generated images are not
   countable scenes**: 72 % of them have objects running off the frame.
8. The prompt template is what decides that. `templates[0]` (the default, and the only one
   evaluation ever used) yields edge-to-edge tiling; `templates[2]`, "… on a white background",
   yields separated countable objects for the same prompt and seed. Training randomises over all
   three templates, so the protocol was also scoring samplers on one phrasing out of three.

**Open, and being run now**
9. Methods A, B and D have not been measured. Three configuration bugs were found and fixed first,
   two of them introduced by this session's own tuning (`tau_q` not rescaled for a 16×16 layer,
   `r_s` below the grid diagonal, `tau_g` ≤ `tau` leaving `L_gap` with no gradient).
10. **Method D** was written in response to finding 4: a count term built on the Euler
    characteristic, which *does* drop when two objects merge. Validated offline (χ equals the true
    count exactly for k = 1…10); not yet trained.

| Date | Who | Run name | Method | Config / overrides | Commit | Steps | Held-out exact acc | MAE | seed ρ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-24 | Atakan | `eval_baseline_owl` | baseline | `configs/base.yaml` + `configs/gpu_8gb.yaml`, `eval.hard_counter=owlv2` | `f9e3559` | – | **0.118** | **4.880** | **0.302** | **The reference for every run below.** Frozen SD-Turbo, 16 seeds × 10 counts × 6 colours × 3 held-out shapes = 2880 images, 122 min. Exact accuracy is 0.642 at n=1 and ≤0.08 from n=3 on. Skill vs the best constant is **−0.95**: the frozen model is roughly twice as wrong as answering "5" every time. Full report: [results/eval_baseline_owl.md](results/eval_baseline_owl.md) |
| YYYY-MM-DD | name | `<run>` | A/B/C | `configs/method_<x>.yaml` | `abc1234` | 6000 | | | | template row |
| 2026-11-24 | Atakan | `smoke_a` | A | `configs/method_a.yaml` + `configs/gpu_8gb.yaml`, `train.steps=20` | `f9e3559` | 20 | – | – | – | GPU bring-up smoke test on RTX 5060 Laptop (8 GB). 0.89 s/step, peak `gpu_mem_gb`=6.61, `skipped`=0, `geom/norm_sq`≈1.003, `geom/d_anchor_max`=0.007 ≪ δ=0.3. `A/k_hat_mean`≈1.0–1.6 at default settings (soft count collapses, see pre-check A) |

## Pre-check results

| Date | Method | Best configuration (layer, t, γ, τ, T / σ, threshold) | MAE or frac ≥ n | Decision |
|---|---|---|---|---|
| 2026-11-24 | A | `up_blocks.1.attentions.1…attn1` (16×16), affinity=attention, t=999, γ=4, τ=0.2, T=0.05 | synthetic MAE 2.30 (constant-predictor bar 2.49 → skill **+0.08**), r=+0.46; generated MAE 9.29 (bar 9.14 → skill **−0.02**), r=+0.21 | **Weak.** k̂ ranks counts on clean synthetic scenes but is badly mis-calibrated (k̂ mean 2.1 vs n mean 5.5), so MAE barely beats a constant guess. No MAE skill on generated. γ=4 is an interior optimum over the union of both scans (γ ∈ {1,2,4,8,16}), so this is not a grid-edge artefact. Do not commit to Method A on this evidence alone |
| | B | counter `runs/counter/counter_best.pt` | | |
| 2026-11-24 | C | `up_blocks.1.attentions.1…attn2` (16×16), t=999, σ=1.0, threshold=0.5 | synthetic r=+0.64, mean peaks 5.8 vs true mean 5.5, MAE 1.95 (bar 2.50 → skill **+0.22**), frac ≥ n = 0.70 | **Best of the two.** Peak count both tracks and is calibrated to n on synthetic. See the note below: ranking by `frac_peaks_ge_n` as `workflow.md` instructs picks the *wrong* configuration |

## Notes

### 2026-11-24 — GPU bring-up and first offline scan (RTX 5060 Laptop, 8 GB)

First runs on GPU hardware; everything before this was CPU-only. Environment: torch 2.11.0+cu128
(Blackwell `sm_120`), driver 582.05, SD-Turbo fp16 weights, `configs/gpu_8gb.yaml` overlay.
Smoke run: 0.89 s/step at `eps_per_batch=1`, `grad_accum=2`, peak 6.61 GB of 7.96 GB.
A full 6000-step run therefore costs roughly 1.5 h on this card.

**Pre-check A.** Scanned γ ∈ {1,2,4,8,16}, τ ∈ {0.05,0.1,0.2,0.3,0.4}, T ∈ {0.01,0.02,0.05},
both affinities, all 16×16 and 32×32 self-attention layers, over two scans
(`runs/precheck`, `runs/precheck_ext`). Best configuration beats a constant predictor by only 8 %
on synthetic and not at all on generated. The correlation is real but the scale is wrong: the soft
count sits near 2 while the true mean is 5.5. Raising γ beyond 4 made it worse, so this is the
honest ceiling for the eigengap read-out as currently specified, not an under-tuned grid.

**Pre-check C — the `frac_peaks_ge_n` criterion is misleading.** `precheck_C_summary.csv` sorts by
`frac_peaks_ge_n`, and workflow.md says to pick the σ and threshold with the best fraction.
Doing that selects σ=0.0, where the detector fires on ~72 local maxima regardless of n:

```
σ=0.0, thr=0.2 : frac>=n = 1.00 for every n, but mean peaks = 72,75,72,75,76,74,76,69,75,73
                 for n = 1..10   -> perfectly flat, r(peaks, n) = -0.07
σ=1.0, thr=0.5 : frac>=n = 0.70, mean peaks = 5.8 vs true mean 5.5, r(peaks, n) = +0.64
```

The criterion is satisfied *most* strongly by the configuration that carries *least* count
information, because over-detection trivially guarantees "at least n peaks". The quantity that
matters is whether the peak count tracks n. Suggested change: rank pre-check C by correlation (or
MAE) against n and treat `frac_peaks_ge_n` only as a necessary-condition filter.

**The colour counter inverts the verdict on the `generated` source — do not use it there.**
`ColorComponentsCounter` defaults (`min_area_frac=5e-4`, one opening iteration) are tuned for clean
synthetic renders. Re-scoring the *same* generator output with OWLv2 (`runs/precheck_owl`):

```
                 OWLv2            colour
overall mean      6.5              15.9        colour over-counts ~2.4x
overall max        28                55
r(found, asked) +0.427            +0.329
```

Carrying that through to the methods, on `generated` at t=999:

| method | with colour counter | with OWLv2 |
|---|---|---|
| A (best γ=4, τ=0.3, T=0.01) | MAE 9.29, bar 9.14, skill **−0.02**, r=+0.21 | MAE 5.24, bar 5.58, skill **+0.06**, r=+0.46 |
| C (σ=1.0, thr=0.5) | MAE 12.97, bar 9.10, skill **−0.43**, r=+0.37 | MAE 4.51, bar 5.60, skill **+0.20**, r=+0.51 |

Both methods look dead on generated data under the colour counter and alive under OWLv2, and the
OWLv2 numbers agree closely with each method's synthetic numbers (A +0.08, C +0.22). The earlier
"no signal on generated" reading was mostly a counter artefact. The frozen model *does* also
over-generate — OWLv2 still finds a median of 21 objects when 10 are asked for — but that effect is
far smaller than the counter's.

**Recommendation:** use a detector counter for the `generated` pre-check source and for any
evaluation of generated imagery. `precheck.hard_counter` and `eval.hard_counter` default to
`color`, which is only appropriate for the synthetic rendered scenes where `n_true` is known by
construction. The cost is modest: the OWLv2 scan took 3m15s against 2m27s for the colour one.

**Two bugs found and fixed during bring-up** (see commit): the repository could not run on Windows
at all (text I/O used the cp1252 locale default, so `configs/base.yaml` itself failed to parse, and
40 of 93 tests failed), and `OwlV2Counter` was broken against transformers ≥ 5, which also disables
the default benchmark counter in `configs/base.yaml`.

### 2026-11-24 — Training plan, and the configuration changes made before training

**Order: C, then A, then B.** Both A and C read the *frozen* model's attention, so each is capped
by how much count information those maps carry — which the pre-check measured as modest. C is
first because its read-out is the better of the two and, more convincing than any single number,
*the same layer wins on both pre-check sources* (synthetic r=+0.64, generated/OWLv2 r=+0.51).
A agrees far less across sources, so its ranking is the less trustworthy of the two. B is last
because it needs `scripts/train_counter.py` run first, but it is the only one of the three not
bounded by frozen-attention quality: it trains its own counter C_ψ. If A and C both land near the
pre-check ceiling, B is where the method still has room.

**Configuration changes, from the scan to the runs.** The pre-check scanned layers and parameters
but nothing wrote the winners back into the configs, so the committed defaults were still the
hand-picked ones. Training against those would have measured the wrong configuration:

| | config default before | set from the scan | evidence |
|---|---|---|---|
| A `self_layers` / `cross_layers` | `up_blocks.2.attentions.1…` (32×32) | `down_blocks.1.attentions.1…` (32×32) | best on generated (MAE 5.24, r=+0.46), near-best on synthetic (2.36 vs 2.30) |
| A `spectral.gamma` | anneal 1.0 → 2.0 | anneal 2.0 → **4.0** | γ=4 is an interior optimum over γ ∈ {1,2,4,8,16} |
| A `spectral.tau` | 0.1 | **0.3** | scan best on the generated source |
| C `cross_layers` | `up_blocks.2.attentions.1…attn2` (32×32) | `up_blocks.1.attentions.1…attn2` (**16×16**) | best on *both* sources |
| C `smoothing_sigma` | 0.0 | **1.0** | σ=0.0 fires ~72 peaks regardless of n (r=−0.07); σ=1.0 gives r=+0.64 |
| C `peak.r_min` / `r_s` | 3.0 / 2.5 tokens | **1.5 / 1.25** tokens | *derived, not scanned* — see below |

Where the two pre-check sources disagreed (Method A's layer), the **generated** source decided:
training reads one-step samples from pure noise, so generated conditions are what the sampler will
actually meet. The synthetic source is a rendering of coloured shapes, useful for checking a
read-out against a count known by construction, but not what the sampler sees.

**One value is inferred rather than measured, and should be treated as provisional.** C's scan
winner is a 16×16 layer, but `peak.r_min` and `peak.r_s` are in *token* units and their committed
values were derived for a 32×32 layer. Following the derivation recorded in the config (a shape
radius of 0.045–0.09 of the image, which is 1.4–2.9 tokens at 32×32), the same physical radius is
0.7–1.4 tokens at 16×16, so both were halved. This is arithmetic from a stated assumption, not a
scanned optimum. The alternative — resampling the 16×16 layer to 32×32 with `attention.grid` and
keeping the original radii — was not tried. If Method C underperforms its pre-check, this is the
first thing to re-examine.

**Checkpoints are written outside the worktree.** `experiment.output_dir` points at the main
checkout's `runs/`, not the worktree's. A git worktree is deleted when it is cleaned up, and a
6000-step run is about 90 minutes of GPU time; writing checkpoints inside one risks losing them to
routine housekeeping. `runs/` is gitignored there as `contributing.md` requires.

**What "saving the U-Net weights" does and does not mean here.** Two networks are involved and
only one of them trains:

| network | parameters | trained? | where it lives |
|---|---|---|---|
| Δφ, the count-conditioned sampler (`CountNoiseSampler`) | **3,202,628** | yes — this is the entire trainable model | `runs/<name>/sampler_step<N>.pt`, `sampler_last.pt` |
| SD-Turbo U-Net | ~865,000,000 | **no, frozen** | re-loaded from the HuggingFace cache |

So there is no trained U-Net to save: the method's whole claim is that the frozen generator is left
untouched and only the noise it is handed changes (Eq. 2, `z_n = ε + w·Δφ(ε, n, c̄)`). Each
checkpoint carries the sampler weights, the optimiser state and the resolved config, so runs are
resumable and self-describing — which is why a 3.2 M-parameter model produces a 38.6 MB file.

**Reporting.** `scripts/report_run.py` generates the full per-run record (environment, resolved
config, checkpoint SHA-256 prefixes, per-stage training curves, complete metrics with the per-count
breakdown and a best-constant-predictor reference bar) into [results/](results/README.md). The
reports are generated, never hand-written, so they cannot drift from what is on disk.

### 2026-11-24 — What the frozen baseline actually does, and what success would look like

Written **before** any trained sampler was evaluated, so the bar is not moved afterwards.

**The frozen baseline is not count-blind — it saturates.** Inspecting the baseline grids
(`runs/figures/baseline_count_response.png`, one seed, n = 1…10, three held-out shapes) shows a
consistent pattern: SD-Turbo places the right number of objects at n = 1, 2 and often 3, then
compresses, and from roughly n ≥ 5 produces 12–16 objects no matter what was asked. The count
response rises and then flattens.

This matters for every number in this log. A flat response and a saturating response can post the
same MAE, so `scripts/report_run.py` now prints the **count-response table** — mean counted value
per requested n, plus the slope of counted against requested — and that table, not the MAE, is the
first thing to read.

**So the project's target is specific.** The sampler does not have to teach SD-Turbo to count from
nothing; it has to extend a response that already works at small n into the range where it
saturates. Success is the slope rising toward 1 and the bias at n ≥ 5 shrinking. An improvement in
overall MAE *without* a change in the response curve is not success: it almost certainly means the
sampler drifted toward the best constant, which on this evaluation (counts uniform on 1…10) is an
MAE of 2.5.

**Pre-registered criteria**, against the frozen baseline measured with the same OWLv2 counter:

| outcome | criterion |
|---|---|
| **works** | response slope rises materially toward 1, `seed_spearman_mean` rises, and skill vs the constant bar is clearly positive |
| **partial** | per-count bias shrinks at n ≥ 5 but the slope barely moves — the sampler compresses less without really counting |
| **no effect** | response curve within noise of the baseline, whatever the MAE does |
| **degenerate** | MAE improves while the slope goes *flatter*, or `diversity` collapses — the sampler found the constant, or a single layout that games the counter |

**One caveat about `nn_regularity_cv`.** `workflow.md` reads a low value as a possible sign of
signal hacking (§8). The baseline grids show near-perfect lattice layouts already — the frozen
model lays objects out on a grid by default on these prompts. Grid-like layout in a trained run is
therefore *not* by itself evidence that the sampler is gaming the counter; only a drop relative to
the baseline is. Compare the number against the baseline, never against zero.

**Some (colour, shape, seed) combinations degenerate into texture.**
`runs/figures/baseline_texture_failure.png` shows two seeds of "n red pentagons" that produce
kaleidoscope and wallpaper patterns rather than countable scenes, at every n. The hard counter
still returns a number for these, and that number is close to meaningless. They are a minority of
the evaluation set but they inflate the counted values at every n, which is part of why OWLv2's
overall mean sits well above the requested mean.

### 2026-11-24 — Frozen baseline measured (`eval_baseline_owl`)

2880 images, OWLv2 counter, 122 min on the RTX 5060 Laptop (2101 ms/image). Full generated report:
[results/eval_baseline_owl.md](results/eval_baseline_owl.md).

| metric | value |
|---|---|
| `overall.exact_acc` | 0.118 |
| `overall.within1_acc` | 0.239 |
| `overall.mae` | 4.880 |
| best-constant MAE (answer "5" always) | 2.500 |
| **skill vs constant** | **−0.952** |
| `seed_spearman_mean` | 0.302 |
| `seed_distinct_counts_mean` | 6.08 |
| `diversity` | 0.461 |
| `nn_regularity_cv` | 0.291 |

**The frozen model is worse than a constant.** MAE 4.88 against a best-constant bar of 2.50. It
over-generates: counted mean 6.51 against a requested mean of 5.50, and it overshoots on 48 % of
images against undershooting on 40 %. So "beat the baseline" and "actually count" are two
different bars, and both should be quoted for every method:

- beat the frozen baseline → MAE < 4.88
- actually count → MAE < 2.50 (skill > 0)

**The count response, measured.** The earlier read from the preview grids was partly wrong, and the
numbers correct it in two places:

| requested n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| counted mean | 1.21 | 3.18 | 6.18 | 6.06 | 7.68 | 8.27 | 8.19 | 7.60 | 8.34 | 8.39 |
| exact acc | **0.642** | 0.191 | 0.083 | 0.045 | 0.049 | 0.017 | 0.031 | 0.062 | 0.031 | 0.028 |

1. The baseline is accurate at **n = 1 only** (64 % exact), not across n = 1–3. By n = 2 it already
   averages 3.18, and by n = 3 it averages 6.18 — it overshoots as soon as more than one object is
   asked for. Slope over n ≤ 3 is **+2.49**, which is over-response, not tracking.
2. The plateau is at **~8 objects**, not the 12–16 that eyeballing a few dense grids suggested.
   Slope over n ≥ 4 is **+0.27**, i.e. nearly flat.

So the shape of the problem stands — tracks at the bottom, saturates above — but the usable range
is narrower than it looked and the saturation level is lower. A method only earns the word
"counting" if it lifts the n ≥ 4 slope.

**A third of seeds do not follow the count at all.** Over the 288 (seed, colour, shape) rows,
Spearman ρ has mean 0.302 but median 0.448, min −0.846, max 1.000, and **32 % of rows have ρ ≤ 0**.
At least one row produced a single distinct count across all ten requests. The §8 question "can a
seed's preferred count be moved?" therefore has a concrete target: move those 32 %.

**Measurement caveat: OWLv2 returns 0 on 17.5 % of images, and the rate is not uniform.**
503 of 2880 images are counted as empty, and the rate depends far more on the prompt's colour and
shape than on anything the generator did:

| | zero-rate | counted mean | | | zero-rate | counted mean |
|---|---|---|---|---|---|---|
| cross | 33.5 % | 3.65 | | blue | **36.9 %** | 2.32 |
| pentagon | 11.0 % | 7.73 | | green | **4.2 %** | 8.69 |
| diamond | 7.8 % | 8.15 | | red | 23.5 % | 8.12 |

A 9× spread in zero-rate between blue and green, and a counted mean of 2.32 for blue against 8.69
for green, is not plausibly the generator making four times fewer blue objects. `OwlV2Counter`
queries `"a photo of a {color} {shape}"` at a fixed `threshold=0.3`; stylised renders (blue
diamonds come out as low-contrast crystals) fall below it consistently. The zero-rate also climbs
with n (14 % at n=1 to 22 % at n=10), as objects get smaller and denser.

**What this does and does not invalidate.** Absolute MAE and exact-accuracy figures carry this
detector bias and should not be quoted as the system's true counting error. The baseline-versus-method
comparison is unaffected: identical prompts, identical seeds, identical counter and threshold, so
the bias is common to both and differences remain meaningful. Worth doing later: sweep
`eval.benchmark.counter_kwargs.threshold`, or report a per-colour breakdown alongside the headline.

**`nn_regularity_cv` baseline is 0.291.** Record it: §8 reads low values as possible signal
hacking, so any trained run must be compared against 0.291, not against zero.

### 2026-11-24 — Method C in training: the stage schedule leaves 4000 steps with no count pressure

Observed live during `c_tuned_s0` and confirmed against the config and the loss code. The run was
**allowed to continue as specified** — this is the method as written, and the as-specified result
is the one worth having. Recorded here so it is not rediscovered later.

**`L_anc` is a hinge, not a pull.** `anchor_loss` is `Σ max(0, d(z_n, ε) − δ)²`, exactly zero while
`d ≤ δ`. At step 420 the run reached `d_anchor_max` = 0.292 against δ = 0.3 with `loss/anc` still
precisely 0.0000. So the sampler drifts freely to the edge of the δ-ball at no cost and only meets
resistance after crossing it. Reaching the edge is not itself a fault — it is the budget being
spent — but a diagnostic that reads a rising `d_anchor_max` as "the anchor is now fighting back" is
wrong, and `loss/anc` must be read alongside it.

**The real problem is where the sampler spent that budget.** At step 450:

| quantity | value | meaning |
|---|---|---|
| `geom/d_anchor_mean` | 0.2684 | how far z_n moved from ε |
| `geom/d_pair_mean` | 0.0405 | how far z_n moved from z_m, different counts, same ε |
| ratio | **0.151** | only 15 % of the movement is count-specific |

Every count is being mapped to nearly the same latent. The sampler has learned a large
count-*independent* shift — which cannot change the count, whatever it does to the loss.

**This follows from the stage schedule, not from bad luck.** `L_ord` is the only term that
explicitly forces z_n to depend on n, by driving `d(z_n, z_m)` toward `α·|log n − log m|` with
α = 2δ/log N_max = 0.2606:

| pair | L_ord target `d(z_n, z_m)` | observed at step 450 |
|---|---|---|
| n=1 vs 2 | 0.1806 | |
| n=1 vs 10 | 0.6000 | mean over sampled pairs **0.0405** |
| n=5 vs 10 | 0.1806 | |

But `configs/method_c.yaml` activates `ord` at step **4000** of 6000. For the first two thirds of
training the only count-dependent pressure is indirect, through the n soft centres inside `L_act`
and `L_supp` — and the cheapest way to reduce `L_supp` (which dominates the total, 3.60 of 3.87 at
step 1) is to flatten the attention map everywhere, which is count-independent. That is consistent
with `C/num_peaks_minus_n` sitting at −1 to −4 while `C/peak_value_mean` stays high near 0.8: the
chosen centres stay strong, the other maxima are being erased, and the count of distinct peaks
falls *below* n.

Method A and B schedule `ord` the same way (A at 4000, B at 3000), so this is a property of the
shared recipe, not of Method C.

**Variant to try if the as-specified runs come out count-independent** (none of these has been run
yet, so none is a result):

1. `--set "method.stages=[{start: 0, terms: [act, supp, anc, pri, ord]}]"` — `ord` from step 0, so
   count-dependence is required from the start rather than retrofitted into the last third.
2. Raise the weight on `ord` relative to `supp`, whose scale dominates the total.
3. `--set geometry.ord_form=rank` — the ranking form (Eq. 6) only asks for the right *ordering* of
   distances, not their absolute scale, which may be easier to satisfy from a cold start.

The quantity to watch is `geom/d_pair_mean / geom/d_anchor_mean`. Near 0 means the sampler is
shifting all counts together and cannot be counting; approaching the `L_ord` targets means the
latents are genuinely separating by count.

### 2026-11-24 — Method C is signal-hacking: it draws tally marks, not objects

Seen in the step-1000 preview of `c_tuned_s0`
(`runs/figures/method_c_signal_hacking_step1000.png`, baseline row above sampler row, seed 1,
counts 1…6, prompt "n red circles"):

| requested n | what the sampler produced |
|---|---|
| 1 | one pink circle — correct |
| 2 | two digit-like glyphs, roughly "9 0" |
| 3 | **three vertical bars** |
| 4 | **four vertical bars** |
| 5 | a cluster of ~15 small blobs |
| 6 | one large amorphous blob |

At n = 3 and n = 4 the model draws the right *number of strokes* and no circles at all. This is
signal hacking in the most literal form available to Method C: `L_act` + `L_supp` ask for n
well-separated maxima in the cross-attention map, and n vertical bars produce exactly n separated
maxima far more cheaply than n distinct circular objects. The loss is satisfied; the image is
wrong. Spec Remark 4 anticipates this ("may find no separate peaks"); what it did not anticipate is
the sampler *manufacturing* separable structure that is not objects.

Two further off-distribution symptoms in the same preview: every sampler image has a blue-violet
border frame, and backgrounds are washed out to near-white, while the baseline row keeps SD-Turbo's
normal textured backgrounds. The latent is being pushed to the edge of the δ-ball
(`geom/d_anchor_mean` ≈ 0.28 against δ = 0.3) and the frozen generator is leaving its training
distribution.

**This corrects the earlier entry's reading.** The previous note concluded from
`d_pair_mean / d_anchor_mean` ≈ 0.15 that the sampler had learned a count-*independent* shift and
therefore "cannot be counting". The images show that is wrong: the outputs differ sharply and
systematically with n (one circle, two glyphs, three bars, four bars). A latent separation of only
0.05 is evidently enough for the generator to produce a very different image — the frozen model
amplifies small changes in z. The correct diagnosis is **count-dependent but degenerate**, not
count-independent. The ratio remains worth logging, but it is a weak proxy: it bounds how much the
*latents* differ, not how much the *images* do, and only the latter matters.

**Consequence for the evaluation.** OWLv2 is asked for "a photo of a red circle". Three vertical
bars should score 0, so this degeneracy will show up as a *worse* MAE than the frozen baseline
rather than a deceptively good one. That is the honest outcome and worth having on record, so the
run is being left to finish and be evaluated rather than killed.

**Why the staged schedule will probably not rescue it.** `rep` (step 2000) pushes soft centres
apart, which tally marks already satisfy — it may even reinforce them. `ord` (step 4000) constrains
distances between latents, not what the image contains. Neither term knows the difference between
"n objects" and "n strokes". Fixing this needs a term that is grounded in object appearance, which
is precisely what Method B's trained density counter provides — further reason to get B run.

### 2026-11-24 — Method C collapses to a single object by step 1500

The step-1500 preview (`runs/figures/method_c_mode_collapse_step1500.png`, same layout as the
step-1000 figure) is worse than the tally marks: for **every** count from 1 to 6 the sampler
produces one red disc on a white canvas, rendered as a framed picture. The images are nearly
identical across n.

So the trajectory over 1500 steps is: baseline → tally marks (step 1000) → one object (step 1500).
The scalars agree, and in hindsight they said it first:

| step | `loss/supp` | `C/num_peaks_minus_n` | `C/peak_value_mean` | `geom/d_pair_mean` |
|---|---|---|---|---|
| 1 | 3.603 | +1.75 | 0.752 | 0.000 |
| 500 | 0.506 | −4.12 | 0.741 | 0.050 |
| 1000 | 0.716 | −2.62 | 0.869 | 0.041 |
| 1600 | 0.238 | −3.75 | 0.915 | 0.055 |

**Why one object is the optimum of stage 1.** The soft-centre repair computes a_k as a
softmax-weighted maximum of ā *inside cluster k*. Nothing requires the n clusters to be on
*different* objects. If all n soft centres land on one broad blob, every a_k is high, so
`L_act = −(1/n)Σ log a_k` is minimised; and there are no other local maxima left, so
`L_supp = Σ_{p∈S} ā(p)²` goes to nearly zero. One large object is therefore a near-global optimum
of the stage-1 objective {act, supp, anc, pri}. `loss/supp` fell 15× and `peak_value_mean` rose to
0.915 — the loss is being minimised exactly as written, while the images get further from the task.

`L_rep` is the only term that forbids coincident centres, and it does not switch on until step
2000. Stage 1 therefore has 2000 steps in which collapse is the cheapest available solution, and
the model reaches it inside 1500.

**This is a concrete, reportable finding about the method as specified**, not a tuning problem: the
differentiable relaxation of "n separated peaks" admits a degenerate optimum that the discrete
formulation (NMS peak counting) does not. `C/num_peaks_minus_n`, which *is* computed by NMS, tracks
the degeneracy correctly throughout — it sits at −2 to −4 from step 200 onward while the optimised
losses keep improving. **A diagnostic computed the discrete way disagreed with the differentiable
loss for 1400 steps, and the diagnostic was right.**

**Suggested change to the staged schedule** (untested): start `rep` at step 0 alongside `act` and
`supp`, so coincident centres are never cheap, rather than at 2000 once collapse has already been
learned. Keeping `ord` late is defensible; keeping `rep` late is what permits this specific
failure.

**Run order changed as a result: B now runs before A.** Methods A and C both read the frozen
model's attention maps, and C has now demonstrated that an attention-based objective can be driven
to a degenerate optimum that leaves the image wrong. Method B's error signal is a trained density
counter applied to x̂0 — the only one of the three grounded in what objects look like rather than in
where attention lands — so it is the most informative next experiment. A is unchanged and runs last.

**Queue and the decision point after B.** Order is now C (running) → B → A, roughly 100 min of
training plus 100 min of evaluation each; the evaluation is the expensive half because OWLv2 costs
2.1 s per image over 2880 images, and the seed count must stay at 16 to stay comparable with the
baseline. Method B's prerequisite is cheaper than expected: 22 000 scenes rendered and VAE-encoded
once into `data_cache/counter/`, then 9360 optimiser steps on a small conv net over cached
latents — tens of minutes, not hours. Its config, model and trainer were checked to import and
resolve before queueing.

After B reports, choose between the two remaining candidates rather than running both blind:

- if **B works**, run A as queued — it completes the three specified methods for the write-up;
- if **B also fails**, A is unlikely to add information (it reads the same frozen attention maps
  that C just gamed, with a weaker pre-check: skill +0.06 against C's +0.20). The better use of the
  slot is the `rep`-from-step-0 variant of C, which tests the specific mechanism this log
  identified instead of confirming a third failure.

### 2026-11-24 — `L_rep` does not break the collapse, and cannot: a correction

Stage 2 activated (logged `stage` 2.0 from step 2010) and the collapse is unchanged. The
step-2000 preview still shows one red disc for every count from 1 to 6. The decisive number:

```
stage 1 (after step 1000):  loss/rep = 0.00000   pks-n = -3.40   d_pair = 0.0646   supp = 0.465
stage 2 (steps 2010-2190):  loss/rep = 0.00000   pks-n = -3.32   d_pair = 0.0669   supp = 0.241
```

**`loss/rep` is exactly zero while it is active.** It is not failing to be optimised — it is
already satisfied. `L_rep = Σ_{i<j} max(0, r_min − ‖m_i − m_j‖)²` is zero as soon as every pair of
soft centres is at least `r_min` = 1.5 tokens apart, and on a 16×16 grid a single blob spanning six
to eight tokens has ample room for ten centres at that spacing. Soft k-means over one broad blob
returns n well-separated centroids, and `L_rep` is content.

**This corrects the recommendation in the previous entry.** That entry proposed starting `rep` at
step 0 to prevent the collapse. It would not have worked: `rep` is satisfied *by* the collapsed
state, so it exerts no gradient at any point in training, early or late. Moving it earlier changes
nothing. The suggestion was wrong and is withdrawn.

**The underlying flaw is in the repair itself, and it is worth stating precisely.** Making the
centres differentiable by soft k-means also detaches them from object identity. The centres are
positions in the attention map, not objects; clustering a single blob into n parts yields n
legitimate centres with no second object anywhere. Every stage-1 and stage-2 term is then
simultaneously satisfiable by one large blob:

| term | what it asks | why one blob satisfies it |
|---|---|---|
| `L_act` | each a_k high | every cluster sits on the blob, so every a_k is high |
| `L_supp` | no maxima far from the centres | there are no other maxima left |
| `L_rep` | centres ≥ r_min apart | k-means spreads them across the blob |

Note this is **not** an artefact of halving `r_min` for the 16×16 layer. The constraint is
scale-invariant: 3.0 tokens at 32×32 and 1.5 at 16×16 are the same fraction of the image and admit
the same number of centres inside a blob of given physical size. The weakness is inherent to
constraining centres rather than objects.

**So what would actually test the mechanism?** Not a schedule change. Either a term that penalises
the spatial *extent* of each cluster — so one blob cannot host n centres — or an error signal
grounded in object appearance rather than in attention geometry. The second is Method B, which is
next in the queue. Note `L_ord` at step 4000 constrains distances between *latents* and will not
address this either, so the remaining 3800 steps are expected to continue in the same regime; the
run is still being left to finish so the comparison table has an as-specified Method C number.

### 2026-11-24 — Method B: what to expect, written before it runs

Recorded in advance so the outcome can be checked against a prediction rather than explained
afterwards. Method C's post-hoc story was convincing partly *because* it was post-hoc; this is the
corrective.

**Why B should do better than C.** Its error signal is `N̂_k = Σ_xy C_ψ(x̂0)_k`, the integrated
density a trained counter assigns to category k. Unlike `L_act`/`L_supp`, which score the geometry
of an attention map, this is grounded in what the decoded image looks like. One large blob is not a
free lunch here: the counter was trained on scenes whose true counts are known by construction, so
a single object should integrate to roughly 1 whatever its size.

**The specific risk, which the spec names.** `density.py` carries Remark 3 — "the sampler can
exploit the counter" — and the setup is exactly a white-box adversarial one: C_ψ is **frozen**,
and the sampler optimises against it through the frozen generator with full gradients. Nothing
stops it finding an x̂0 that drives the density sum to n without containing n objects. That is the
same failure as Method C's, one level up: not gaming attention geometry, but gaming a learned
critic.

**Three defences B has that C did not:**

1. `n_k = 0` for every category the caption does not mention, over all 48 (colour, shape) channels.
   Stray mass anywhere is penalised, so the cheap solutions available to C are costly here.
2. Heavy augmentation during counter training (`online_aug` p=0.8 with contrast, blur and noise;
   `image_aug` p=0.5 before VAE encoding). A counter that survives that is harder to fool with a
   small perturbation.
3. The evaluation counter is **OWLv2, not C_ψ**. If B games its own counter, the evaluation says
   so. This is Remark 3's prescribed check and it is already wired into the queued run.

**Predictions, in order of expected likelihood:**

| outcome | what it would look like |
|---|---|
| counter transfer fails | pre-check B's `generated` rows much worse than `synthetic` — the synthetic-to-x̂0 gap of §4.1. Fix is `scripts/harvest_generator_samples.py` and retrain |
| adversarial exploitation | `B/count_abs_err` falls nicely during training while OWLv2 exact accuracy at evaluation does not improve, or falls |
| genuine partial success | both improve, most likely at small n where the frozen baseline already has a response to amplify |

**The diagnostic to watch** is the gap between `B/count_abs_err` (the frozen counter's own view,
logged every step) and the OWLv2 result at evaluation. With Method C the discrete NMS diagnostic
disagreed with the differentiable loss for 1400 steps and the diagnostic was right; here the same
role is played by the independent counter, and the same rule applies — when the trained critic and
the independent check disagree, believe the independent check.

### 2026-11-24 — Method C, stage 3: every term succeeds and the task still fails

`L_ord` activated at step 4010 and **worked, quickly**:

```
stage 2 (200 logs):  loss/ord = 0.2281   d_pair = 0.0648   pks-n = -3.39
stage 3 ( 24 logs):  loss/ord = 0.0216   d_pair = 0.2079   pks-n = -3.54
```

Within ~240 steps `loss/ord` fell more than tenfold and `d_pair_mean` rose from 0.065 to ~0.21,
which is the range `L_ord` actually asks for (α·|log n − log m| runs from 0.18 for near pairs to
0.60 for n=1 vs n=10). The latents genuinely separate by count now. `geom/d_anchor_mean` sits at
0.29–0.31, so the anchor hinge is finally active and binding.

**And `C/num_peaks_minus_n` did not move at all: −3.39 before, −3.54 after.** The step-4000 preview
(`runs/figures/method_c_ord_active_step4000.png`) shows why: the images do now differ with n — a
thin ring at n=1, a disc at n=2, then a fused peanut shape that morphs through n=3…6 — but every
one of them is a *single* object. Count-dependence arrived and expressed itself as reshaping one
blob.

**This completes the Method C result, and it is the interesting part.** Taken term by term, the
method works exactly as designed:

| stage | term added | its own objective | outcome |
|---|---|---|---|
| 1 | `act`, `supp` | strong peaks, no spurious maxima | **achieved** — `loss/total` 3.872 → 0.125 |
| 2 | `rep` | centres ≥ r_min apart | **already satisfied**, `loss/rep` = 0.00000 throughout |
| 3 | `ord` | d(z_n, z_m) ∝ \|log n − log m\| | **achieved** — ord 0.228 → 0.022, d_pair 0.065 → 0.208 |

Every term reaches its stated goal. The conjunction still produces one object for every count,
because **no term in the set requires the image to contain n objects**. `act` and `supp` constrain
the shape of an attention map, `rep` constrains distances between inferred centres, `ord`
constrains distances between latents. Each is a proxy; the sampler satisfied all three proxies
without touching the quantity they stand in for.

That is a cleaner negative result than "training diverged", and it is the one to report: the
failure is in the objective, not the optimisation. The optimiser did its job well — `skipped` = 0
for the whole run, `geom/norm_sq` ≈ 1.00 throughout, gradient norms falling smoothly.

### 2026-11-24 — Full training trajectory, and a second failure: colour drift

Two compiled figures, one row per preview step, one column per count, with the frozen baseline on
top: `runs/figures/method_c_training_trajectory.png` ("n red circles") and
`runs/figures/method_c_trajectory_pentagon.png` ("n blue pentagons").

**The trajectory.** step 0 is pixel-identical to the frozen baseline, which confirms empirically
that the zero-initialised `conv_out` gives z_n = ε at initialisation. Then:

| step | what the sampler produces |
|---|---|
| 500 | near-total destruction — washed-out white fields inside a saturated blue border, faint ghost outlines |
| 1000 | structure returns: count-dependent tilings (a 2×2 grid at n=4, 3×3 at n=6), still blue-framed |
| 1500 | collapse to one object for every n ≥ 2 |
| 2000–3500 | one object, drifting in shape and colour |
| 4000 | `L_ord` active: shapes morph with n (ring → disc → peanut) but remain a single object |

The step-500 row is worth keeping: the first thing the sampler learns is to wreck the image. That
it recovers structure by step 1000 says the anchor and prior terms do their job — but it spent
~500 steps outside anything resembling the data distribution.

**Colour drift, which the count metrics would never have caught.** In the pentagon figure the
requested colour is **blue** at every step. The sampler's output goes blue → washed-out → pink →
mauve → purple → dark purple → **red** by step 4000. The prompt asks for blue pentagons and the
model delivers red ones.

Nothing in Method C's objective constrains colour. `L_act` and `L_supp` score the geometry of one
cross-attention layer; `L_rep` scores centre separation; `L_ord` scores latent distances; `L_anc`
and `L_pri` only bound how far z may move and keep its norm near 1. Colour fidelity is free to be
traded away for peak structure, and it was. This is the same lesson as the count failure in a
second dimension: **the method constrains proxies of the image, so everything about the image that
no proxy mentions is unprotected.**

**Prediction for the evaluation, recorded now.** OWLv2 is queried with `"a photo of a blue
pentagon"`. Red pentagons should return no detections at all, so Method C's evaluation is expected
to be far worse than the frozen baseline, driven not only by wrong counts but by near-zero
detection rates — a much larger share of zero counts than the baseline's 17.5 %. If instead C's
counts come out merely mediocre, that would mean the colour drift seen in previews does not carry
to the evaluation prompts, and the preview would need re-examining.

**Statistical note: seed variance is large, so comparisons must be paired.** The baseline's own
first and last eight seeds give MAE 4.44 vs 5.32 and n ≥ 4 response slopes of +0.42 vs +0.13 — a
threefold spread in the headline slope from seed choice alone. An 8-seed estimate of MAE has a
bootstrap sd of 0.17 and a 95 % range of [4.51, 5.17]. Two consequences, both now acted on:

1. `eval.seeds` stays at **16**; halving it to save evaluation time would inject noise comparable
   to any effect worth detecting.
2. `scripts/compare_paired.py` matches records on (colour, shape, seed, n) and reports the mean
   change in absolute count error with a bootstrap CI resampled **over seeds**, since records
   sharing a seed share a noise vector and are not independent. Unpaired comparison of marginal
   means, which `compare.py` does, is not sensitive enough here and should not be the basis for a
   verdict.

### 2026-11-24 — Correction: the collapse is a local minimum behind a barrier, not the optimum

An audit of the loss code against the specs, plus direct probes of `peak_losses` at the shipped
settings, correct the earlier reading in this log. Loss of the three method terms for n=3 against
the number of objects actually present (16×16 map, σ=1 blobs, `tau_q`=1.0, `r_s`=1.5):

| objects k | 1 | 2 | **3** | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| `L_act` | 0.184 | **2.353** | 0.166 | 0.038 | 0.073 | 0.137 | 0.097 | 0.080 |
| `L_supp` | 0.000 | 0.000 | 0.000 | 3.000 | 3.520 | 7.587 | 5.598 | 6.280 |
| **total** | 1.184 | 2.353 | **0.166** | 3.038 | 3.593 | 7.724 | 5.695 | 6.360 |

**The correct layout is the global minimum**, seven times better than the collapsed one. The
earlier entries called collapse "a near-global optimum of the stage-1 objective"; that was wrong.
What is true is narrower and more useful:

1. **Collapse is a local minimum.** From k=1 (1.184) both neighbours are uphill: k=2 costs 2.353
   and k=4 costs 3.038. Once there, gradient descent cannot leave.
2. **The barrier is the farthest-point fill.** At k=2 with n=3, `init_centers` has only two local
   maxima to use and places the third centre by farthest-point selection into empty space, where
   a_k ≈ 0.5 and `L_act = −(1/n)Σ log a_k` explodes to 2.353 — fourteen times its value at the
   correct layout. Having *fewer* objects than requested is punished far harder than having one.
3. **The asymmetry points at collapse.** `L_supp` punishes extra peaks brutally (3.0 at k=4, 7.6 at
   k=6) while under-counting costs almost nothing (`L_supp` = 0 at k=1). The frozen baseline
   over-generates — about 8 objects when 3 are asked — so training starts at k≈8 (loss 5.695) and
   *both* directions are downhill: toward k=3 (0.166) and toward k=1 (1.184). The route to k=1 is
   the easy one, because suppressing everything needs no selectivity, and it is a trap once reached.

That is a materially different claim from "no term requires n objects". The terms do encode n
objects; the landscape between over-generation and the right answer is what defeats them.

**Three configuration bugs, two of them introduced by the tuning committed in `d8b3837`:**

| config | was | now | why |
|---|---|---|---|
| `method.peak.tau_q` (C) | 4.0 | **1.0** | τ_q is in tokens² (`peak.py`: `−d2/tau_q`). Halving lengths for the 16×16 layer required *quartering* it. At 4.0 the assignment kernel is √(τ_q/2)=1.41 tokens wide — wider than the `r_min`=1.5 separation it must enforce, so soft k-means cannot hold two centres apart |
| `method.peak.r_s` (C) | 1.25 | **1.5** | local maxima come from 3×3 max-pool, whose diagonal neighbour is at √2=1.414 tokens. Below that a centre does not cover its own neighbourhood, so an object's own peak enters the suppression set and `L_supp` pushes it down while `L_act` pushes it up |
| `method.spectral.tau_g` (A) | 0.2 | **0.45** | must exceed `tau` plus a few `T`. Measured with the real `soft_count`/`gap_loss` at n=3, τ=0.3, T=0.02: at λ=0.20 `L_gap` is already exactly 0 while the soft count still reads **3.993**. The entire useful range λ ∈ [0.20, 0.34] received no gradient |

The first two came from halving C's radii for the 16×16 layer without rescaling the quantities that
depend on them; the third from taking τ=0.3 off the scan without re-checking τ_g against it.
Method A had not yet run, so its fix lands before the run rather than after.

**One claim not reproduced.** The audit reported that `tau_q`=1.0 deepens the correct minimum about
thirtyfold. Probing the real `peak_losses` with planted Gaussians, the collapsed-minus-correct
margin moves only from 1.1145 to 1.1469 (+3 %). The two probes construct the collapsed scene
differently and the discrepancy is unresolved. The `tau_q` change is kept because the units
argument is sound on its own, but **no measured thirtyfold improvement should be quoted.**

### 2026-11-24 — The evaluation was measuring the counter, not the generator

An audit of the evaluation protocol against `runs/eval_baseline_owl` found the instrument
unreliable enough that **no conclusion drawn from it about n ≥ 5 is safe.** Verified directly
against the 2880 baseline records:

**The detector query matches proper nouns.** Zero-detection rate by prompt:

| prompt | blue cross | red cross | orange cross | purple cross | green cross | yellow cross |
|---|---|---|---|---|---|---|
| returns nothing | **90.0 %** | **70.0 %** | 19.4 % | 13.1 % | 6.9 % | 1.9 % |
| mean count | 0.11 | 0.45 | 4.21 | — | — | — |

The two catastrophic cells are exactly the two phrases that name an organisation — *Blue Cross*,
*Red Cross*. `detector_query` builds `"a photo of a blue cross"`, whose text embedding is dominated
by the emblem rather than by a blue cruciform object. The audit established the zero images are
not empty: 99.2 % of them have the target colour over ≥10 % of the image, mean colour coverage in
zero cells (43.5 %) is indistinguishable from non-zero cells (47.7 %), and 459 of 503 contain three
or more separate colour components. The detector is failing, not the generator.

**At n ≥ 5 the measurement is at chance.** Exact accuracy 3.65 % against 3.72 % for drawing from
the detector's own observed count distribution. The two counters available in this repository agree
on these images with correlation 0.131.

**Three repairs, now the default in `configs/base.yaml`:**

| setting | was | now | why |
|---|---|---|---|
| `threshold` | 0.3 | **0.1** | 0.3 is three times the library default and the model card's value; counts flicker 0↔11 between visually near-identical adjacent-n images, the signature of sitting on a threshold |
| `query_color` | colour in the query | **false** | removes the proper-noun confound; colour fidelity is better measured separately, which the colour-drift finding makes necessary anyway |
| centre coordinates | `x/w, y/h` | `x/side, y/side` | boxes return in padded-square pixels, so both axes normalise by the longer side. Harmless on the 512×512 synthetic set, wrong on the benchmark photos |

**Consequence: the baseline and Method C are being re-measured.** `eval_baseline_owl` and the
numbers quoted from it earlier in this log are retained as a record of the old protocol, but
`eval_baseline_fix` supersedes them and every method is compared against that. Method C's
evaluation was stopped two grids in rather than spend two hours on the broken instrument.

**Not changed, and worth a separate experiment.** `PromptSampler.prompt` always uses
`data.templates[0]`, the bare noun phrase `"{count} {color} {shape}"`, so SD-Turbo answers with
edge-to-edge decorative tiling: 72.4 % of baseline images have the target colour touching the
border, meaning objects run off the frame and the true count is not even well defined.
`data.templates[2]` (`"… on a white background"`) exists and is never used. Switching it would
likely be the single largest change to MAE of anything identified — but it changes the *task*
rather than fixing the instrument, so it belongs in its own clearly-labelled run, not folded
silently into the protocol.

**Also deferred, with the evidence recorded:** `nn_regularity_cv` is not count-invariant (its
uniform-random null rises from 0.342 at k=3 to 0.555 at k=30, so it mostly reads how many boxes the
detector produced) and is available on only 58 % of records, skewed by colour (blue 26 %, green
77 %). It should be replaced by a null-corrected statistic such as Clark–Evans before being quoted.

### 2026-11-24 — Correction: batching the counter bought ~20 %, not 2–3×

Measured against the first grids of `eval_baseline_fix`: **5.4 min per grid of 160 images, against
6.8 min for the original unbatched `eval_baseline_owl` run** — a 20 % saving, and the full
evaluation falls from about 122 min to about 97 min. The commit message for `32ae566` framed
evaluation as dominated by one-forward-per-image detector calls; that framing was wrong.

The reasoning error is worth recording because it is a general one: **batching a vision transformer
amortises kernel-launch and Python overhead, it does not reduce FLOPs.** OWLv2 processes a
960×960 input per image whether it is called 160 times or 20 times with batches of 8, so the GPU
work is unchanged and only the per-call overhead disappears. The optimisation is still correct and
worth keeping — counts are identical, verified at batch 2/4/8 — but it is a 20 % saving, not a
structural one.

Two other candidates were measured and ruled out rather than assumed:

- `_nms` is a pure-Python O(n²) loop, so lowering the threshold from 0.3 to 0.1 (which returns far
  more boxes) was a plausible regression. Measured: 10.5 ms at 100 boxes, i.e. 1.7 s per 160-image
  grid against 324 s total. Not material.
- Model loading accounts for roughly the first 2–3 minutes of a run, which is why the first grid
  always looks slower than the rest.

The real cost is the detector's forward pass at 960×960 plus the VAE decode at 512×512, and
neither is reducible without changing the detector or the number of images. Since seed variance
forces `eval.seeds` to stay at 16 (the baseline's own halves differ by 0.88 MAE), **about 97 min
per evaluation is simply the price**, and the pipeline is planned around that rather than around
a hoped-for speedup.

### 2026-11-24 — Method D: a count objective that notices a merge

Written in response to the Method C result, and **not yet run** — this entry records the design
and its offline validation only.

**The invariant Method C was missing.** Every term it optimises is unchanged when two objects
fuse: centre positions, centre distances, and soft peak values inside a cluster all survive a
merge, so one blob hosting n soft centres satisfies `L_act`, `L_rep` and `L_supp` at once. A count
objective needs a quantity that strictly *decreases* on a merge. The Euler characteristic of a
superlevel set is one, and it is free to compute:

```
chi = V - E + F   over the pixel cubical complex  =  #components - #holes
```

Verified on planted Gaussians at a hard threshold: **χ equals the object count exactly for
k = 1, 2, 3, 4, 5, 6, 8, 10**, and merging two blobs takes χ from 2 to 1 at the moment they touch.

**Three terms, each closing a hole the others leave open:**

| term | form | what it forbids |
|---|---|---|
| `L_cc` | (log(1+χ) − log(1+n))² | wrong count. Log-ratio, so over- and under-counting cost the same — `L_supp`'s asymmetry is what pointed C at collapse |
| `L_area` | relu(min_area − mass/χ)² | n single-token specks. At low occupancy χ ≈ HW·u, so a sparse field reaches any moderate count with nothing in the image. Measured: a speck field gives 1.09 tokens per component against 7.75 for real objects |
| `L_bin` | mean(u(1−u)) | u drifting away from {0,1}, where χ stops meaning what it says |

**The calibration was measured, and the first attempt was wrong.** A threshold relative to the
map's *mean* gave χ = 3.2 for ten real objects while a structureless haze reached 11 — the
degenerate solution sitting at an occupancy where `u(1−u)` is small, so the binarisation term never
saw it. Thresholding on the map's **max** removes the hole at its source, because
`object_attention_map` min-max normalises so the max is a stable reference while the mean moves
with how much of the map is foreground. At `tau_frac` 0.5 and `temperature` 0.02 the soft χ matches
the discrete χ within 0.5 for every k tested, and a uniform haze now saturates to a single
component and is rejected by `L_cc` itself. The parameter sweep is in `tests/test_components.py`
as assertions rather than in a notebook.

**The count term is not staged.** Method C spent its first 2000 steps with nothing in the active
set requiring count-dependence, and had learned a degenerate solution before `L_ord` switched on.

**Declared limitation.** Method D still reads *attention*, so n strokes that draw n attention
components satisfy it exactly as well as n objects do — the tally-mark regime Method C passed
through at step 1000 remains reachable. Closing that needs a term grounded in the decoded image,
which is Method B's territory. Every soft term ships with its discrete counterpart
(`D/chi_hard`, `D/chi_soft_minus_hard`) because with Method C the discrete diagnostic disagreed
with the differentiable loss for 1400 steps and the diagnostic was right.

### 2026-11-24 — The counter repair, validated before the full re-run finished

The frozen generator is deterministic given a seed, so the old baseline's saved grids are the same
images the re-run is producing. Re-counting them with the repaired counter (CPU, 3 seeds × 10
counts per cell) isolates the counter change exactly — same images, same seeds, only the
measurement differs:

| cell | old zero-rate | new zero-rate | old mean | new mean | old MAE | new MAE |
|---|---|---|---|---|---|---|
| blue cross | 93.3 % | **3.3 %** | 0.07 | 8.63 | 5.43 | **3.60** |
| red cross | 50.0 % | **0.0 %** | 0.97 | 9.40 | 4.53 | **3.97** |
| green cross | 0.0 % | 0.0 % | 6.57 | 8.73 | 2.20 | **3.37** |

**The proper-noun catastrophe is gone.** "blue cross" went from returning nothing on 93 % of
images to 3 %, and its mean count from 0.07 to 8.63 — on images that always contained roughly nine
crosses. That cell is now measuring the generator instead of the phrase.

**And MAE got worse where the counter already worked.** Green cross, which had no zero problem,
moved from 2.20 to 3.37. This is the predicted direction and it is not a regression: at threshold
0.1 the detector finds more of the objects that are genuinely there, and since the generator
over-generates, finding more of them pushes the count further above n. The old number was low for
the wrong reason — objects were being missed in a way that happened to cancel part of the
over-generation error.

**So the repair should not be reported as an accuracy improvement.** Overall MAE may move either
way depending on the mix of cells; the three sampled here give −1.83, −0.56 and +1.17. What the
repair buys is a measurement that means the same thing for every prompt, which is the precondition
for comparing methods at all. The earlier `eval_baseline_owl` numbers were an average over cells
measuring different things, one of which was measuring an insurance brand.

**Note for the write-up:** the new means (8.6–9.4 against a requested mean of 5.5) are a cleaner
statement of the over-generation problem than anything in the old protocol, where detector
failures were masking it.

### 2026-11-24 — How good is the ruler? Counter accuracy on images with known counts

`ShapeRenderer` produces scenes containing exactly n objects by construction, so they give a
ground truth the generated set cannot. Six scenes per count across four colours and the three
held-out shapes, n = 1…8, counted three ways:

| counter | MAE | exact |
|---|---|---|
| old OWLv2 (threshold 0.3, colour in query) | 0.548 | 78.6 % |
| **repaired OWLv2** (threshold 0.1, shape only) | **0.143** | **85.7 %** |
| `ColorComponentsCounter` | **0.000** | **100 %** |

Three things follow.

**The repair is a real improvement, not just a re-labelling.** MAE falls 3.8× on images whose count
is known. The earlier caveat stands — it does not follow that MAE on *generated* images improves —
but the instrument itself is measurably better.

**The measurement noise floor on clean scenes is ~0.14 MAE.** Any claimed effect smaller than that
is not distinguishable from the counter, before seed variance is even considered. (Seed variance
is much larger: the baseline's own halves differ by 0.88.)

**And the decisive one: both counters are near-perfect here and disagree wildly on generated
imagery** — correlation 0.131, with the colour counter over-counting 2.4×. A counter that scores
100 % on rendered scenes and is unusable on generated ones is not a broken counter. It means the
generated images are not countable scenes. This matches the border statistic: 72.4 % of baseline
images have the target colour touching the frame, so objects run off the edge and the true count
is not defined, whatever counts.

**This reframes the prompt-template question from a tuning choice into the central one.** The
evaluation uses `data.templates[0]`, the bare noun phrase, which SD-Turbo answers with edge-to-edge
tiling. `data.templates[2]` asks for objects "on a white background" — the same kind of scene the
renderer makes and both counters count perfectly. If that template produces countable images, the
evaluation becomes well posed and every method can be measured properly; if it does not, no
objective and no counter can rescue the protocol, and that is itself the finding.

Queued as its own run. Note the comparison must be baseline-to-baseline on the *same* template:
switching templates changes the task, so numbers across templates are not comparable.

### 2026-11-24 — Prompt template probe: templates[2] does produce countable scenes

A CPU probe (SD-Turbo in float32, ~4 s per image, no GPU contention with the running evaluation)
generated the same prompts and seeds under `data.templates[0]` and `data.templates[2]`.
Figure: `runs/figures/template_probe2.png`, rows = template 0 seeds 0–1 then template 2 seeds 0–1.

| prompt | templates[0] "three green crosses" | templates[2] "… on a white background" |
|---|---|---|
| three green crosses | crosses fused into an edge-to-edge lattice on textured ground | **three separated crosses on white** |
| two yellow diamonds | a field of gem facets filling the frame | **exactly two gems on white** |
| three red circles | overlapping discs running off every edge | discs on white, still over-generated |

**The first probe said the opposite and was wrong.** It used "blue pentagons" only, which tiles
under every phrasing, and concluded that no template helps. That is the second time in this
session a single (colour, shape) cell has produced a confident wrong reading — the first was "red
pentagon" for the baseline count response. Prompts in this set do not behave alike, and no
conclusion should be drawn from one of them.

**Why this outranks the method queue.** Both counters score ~0 MAE on countable scenes and agree
at r=0.131 on tiling, so the template decides whether anything here is measurable at all. Running
four methods against templates[0] would produce a consistent comparison on a task where the
quantity being compared is undefined for three quarters of the images.

**Queue reordered.** The master pipeline keeps the two stages already paid for — the
repaired-counter baseline, and Method C measured against it — and is then stopped before Method B
so the template comparison can run. Which template the remaining methods train and evaluate on is
a decision to take on those numbers, not in advance.

### 2026-11-24 — The "saturation at ~8 objects" was substantially a counter artefact

A CPU count-response probe (n = 1…8, four prompts × two seeds, repaired counter, no GPU
contention) against the figure this log has quoted since the baseline landed:

| measurement | counter | slope (n ≥ 4) |
|---|---|---|
| `eval_baseline_owl`, as reported | old (threshold 0.3, colour in query) | **+0.273** |
| same records, zero-detections excluded | old | **+0.479** |
| CPU probe, templates[0] | repaired | **+1.000** |
| CPU probe, templates[2] | repaired | **+1.925** |

**Mechanism.** The old counter's zero-rate climbs with the requested count — 14.2 % at n = 1,
20.1 % at n = 7, 22.2 % at n = 9 and 10 — because denser scenes with smaller objects fall below a
threshold set three times too high. Dropped detections at the top of the range pull the fitted
response flat. Simply excluding the zeros from the *same* records raises the slope by 75 %, and
measuring properly with the repaired counter raises it further.

**What this revises.** The claim that the frozen model "tracks the count to n ≈ 3 and then
saturates at about 8 objects" is the single most repeated finding in this log, and it is now
suspect. The frozen generator appears to track the requested count considerably further than the
broken instrument suggested; what it does is **over-generate roughly in proportion** (counted mean
9.9 against a requested mean of 4.5 on the probe) rather than flatten. Over-generation with a
slope near 1 is a different problem from saturation, and a more tractable one — a multiplicative
bias can be corrected where a flat response cannot.

**Not yet settled, and deliberately not rewritten in the summary until it is.** The probe is 64
images per template on CPU in float32, against 2880 on GPU in bfloat16 for the real run, and it
covers n = 1…8 rather than 1…10. `eval_baseline_fix` is in progress and will give the definitive
count-response curve with the repaired counter. The honest position until then: the old slope of
+0.273 is an artefact of the measurement, the true value is higher, and the magnitude is pending.

**Methodological note.** Every method result so far was scored against a baseline whose response
curve was distorted this way. That is the third finding in this session where the instrument,
rather than the model, produced the headline number — after the proper-noun zero-rate and the
counter/generated-image disagreement. The pattern is consistent enough to be worth stating as a
rule for the write-up: **on this task, validate the measurement against known counts before
reporting anything derived from it.**

### 2026-11-24 — Baseline re-measured (`eval_baseline_fix`): the model does track the count

2880 images, same frozen model, same 16 seeds, same prompts as `eval_baseline_owl`. **Only the
counter changed.** Figure: `runs/figures/baseline_repaired.png`.

| metric | old protocol | repaired | change |
|---|---|---|---|
| zero-detection rate | 17.5 % | **1.0 %** | −16.5 pts |
| **slope (n ≥ 4)** | +0.273 | **+1.031** | **+0.758** |
| slope (all n) | +0.696 | +1.643 | +0.947 |
| Pearson r | +0.310 | +0.529 | +0.218 |
| `seed_spearman_mean` | 0.302 | **0.667** | +0.365 |
| counted mean | 6.51 | 12.89 | +6.38 |
| MAE | 4.880 | **8.166** | +3.286 |
| skill vs constant | −0.952 | **−2.266** | −1.314 |

**The saturation was the instrument.** The correction predicted from the CPU probe is confirmed at
full scale: the response slope above n = 4 goes from +0.27 to **+1.03**, which is essentially
perfect marginal tracking — each additional requested object yields one additional generated
object. The old counter's misses rose with n (22.2 % zeros at n = 9–10 against 0.0 % now), and
that is exactly what pulled the old curve flat. `seed_spearman_mean` more than doubling says the
same thing independently.

**What the frozen model actually does**, for n ≥ 4:

```
counted = 1.031 · n  +  8.51
```

Slope one, offset **+8.5 objects**. It is not failing to count; it is counting and then adding a
large near-constant surplus. That is a different and far more tractable problem than a flat
response: a bias can be removed, a dead signal cannot.

**MAE got much worse, and that is correct.** 4.88 → 8.17, skill −0.95 → −2.27. The old number was
low because the counter was missing objects that were always there, and those misses partly
cancelled the over-generation. Two errors in opposite directions reading as accuracy. The repaired
number is larger and *true*.

**How much of the error is calibration:**

| | MAE |
|---|---|
| frozen baseline as measured | 8.166 |
| subtract the fitted offset (−8.51) | 6.317 |
| rescale counts to match the requested mean (×0.43) | **2.420** |
| best constant predictor | 2.500 |

A single global multiplicative rescale takes the baseline from skill −2.27 to **+0.03** — just past
the constant bar. So essentially the whole error is a calibration surplus sitting on top of a count
signal that already tracks.

**Consequence for the methods.** The target is now specific and modest: remove a multiplicative
over-generation of roughly 2.3× while preserving a slope that is already 1.0. Every method result
measured before this point was scored against a baseline whose response curve was distorted, and
the pre-registered criteria — written around "lift the n ≥ 4 slope" — need restating: **the slope
is already there. The thing to fix is the surplus.** A method that improves MAE while flattening
the slope would now be a clear regression.

**Caveat.** MAE ≈ 2.42 after perfect global calibration is the ceiling for a *global* correction,
and it barely clears the constant bar, because per-image variance around the mean is large. A
sampler that only rescales the whole distribution gains little; what would pay is reducing the
spread at fixed n.

### 2026-11-24 — Ceiling analysis: the error is half variance, and centring alone cannot win

With a trustworthy baseline it is finally possible to ask what *any* method could achieve on this
generator. Decomposing the baseline's error at each requested count:

| n | counted mean | bias | sd | MAE |
|---|---|---|---|---|
| 1 | 2.57 | +1.57 | 2.92 | 1.63 |
| 3 | 10.23 | +7.23 | 5.60 | 7.33 |
| 5 | 14.34 | +9.34 | 8.00 | 9.82 |
| 7 | 15.74 | +8.74 | 8.49 | 9.85 |
| 10 | 19.12 | +9.12 | 9.28 | 10.95 |

**Bias and spread are the same order of magnitude.** That is the fact the whole project turns on.
Simulating a method that centres the distribution perfectly at each n and shrinks the spread by a
factor s:

| s | MAE | skill |
|---|---|---|
| 1.0 — perfect centring, spread untouched | 5.459 | **−1.184** |
| 0.8 | 4.368 | −0.747 |
| 0.6 | 3.276 | −0.310 |
| 0.5 | 2.730 | −0.092 |
| **0.4** | 2.184 | **+0.126** |
| 0.3 | 1.638 | +0.345 |

**A method that removes the entire +8.5 surplus and does nothing else scores −1.18, worse than
answering "5" to everything.** The spread must fall to roughly **40 %** of its current value before
skill turns positive at all. This is the quantitative statement of why the task is hard, and it
was invisible under the old counter.

**It also explains the one correction that did work.** The global ×0.43 rescale reached skill
+0.032 not by centring — it leaves the mean at 6.1 when 5 is asked — but because multiplying the
counts by 0.43 shrinks the spread by 0.43 as well. It beat the bar through variance reduction,
achieved incidentally.

**The success criterion, restated for the third and hopefully last time.** The pre-registered
version asked for the n ≥ 4 slope to rise; the slope is already +1.03. The revision after the
baseline asked for the surplus to be removed; that alone scores −1.18. The target is:

> reduce the conditional spread of the count at fixed n to under ~40 % of the frozen model's,
> while keeping the response slope near 1 and the mean near n.

Equivalently: make the generator *consistent*, not merely unbiased. `seed_spearman_mean` (0.667 on
the baseline) and the per-n sd are the quantities to watch; MAE on its own will keep misleading.

**Caveat on the degenerate row.** A per-(n, colour, shape, seed) calibration scores MAE 0.000, but
those four fields uniquely identify a single record, so each "group" has one element and the result
is vacuous. It is recorded here only so nobody later mistakes it for a meaningful ceiling.

### 2026-11-24 — Is the variance learnable? A held-out test, and a trap I walked into

**First, the mistake.** The obvious way to ask how much of the count variance the sampler could
correct is to group records by everything the sampler observes — (n, colour, shape, seed) — and
measure the within-group spread. That returns "100 % of variance explained, MAE 0.000", which is
meaningless: those four fields uniquely identify a single record, so every group has one element.
This log had already flagged exactly that degeneracy one section earlier, and the analysis walked
into it anyway. Recorded because the number looks like a spectacular result and is worth nothing.

**Done properly**, by fitting a correction on seeds 0–7 and scoring it on seeds 8–15, so no group
can be a single record and the question becomes what a method could *learn* rather than memorise:

| correction, fitted on seeds 0–7, scored on seeds 8–15 | MAE | skill |
|---|---|---|
| none | 8.972 | −2.589 |
| global affine inverse of `counted = a·n + b` | **3.849** | −0.540 |
| per-(colour, shape) affine inverse | 10.730 | **−3.292** |

Two conclusions, both unwelcome and both useful.

**A global correction generalises, and is still not enough.** Inverting the fitted response more
than halves the error on unseen seeds, 8.97 → 3.85, and remains well short of the 2.50 constant
bar. This is consistent with the ceiling analysis: the correction fixes bias, and bias is only half
the problem.

**Per-prompt correction overfits badly — worse than doing nothing.** The marginal variance table
below makes colour and shape look like strong, usable structure (at n = 5, shape accounts for sd
4.34 of a total 8.00, more than seed's 2.23), but fitting on half the seeds and testing on the
other half gives MAE 10.73 against 8.97 for no correction at all. Those per-prompt effects are
largely sampling noise at this many records per cell, not a stable property of the prompt.

| n | total sd | by seed | by colour | by shape |
|---|---|---|---|---|
| 1 | 2.92 | 0.95 | 0.52 | 0.93 |
| 5 | 8.00 | 2.23 | 2.87 | 4.34 |
| 10 | 9.28 | 2.07 | 4.15 | 0.79 |

**Design consequence.** A method that conditions strongly on the prompt to correct counts is
fitting noise. The usable signal in this decomposition is global, and global correction caps out
below the constant bar. What remains is genuine per-image variance reduction — making the generator
produce a more consistent number for a given (ε, n) — which is exactly the thing none of the four
methods' objectives measures.

**Caveat on the analogy.** These corrections are applied to counts after the fact; a sampler
instead changes what is generated. The numbers bound the structure of the error, not literally what
a sampler can reach. A sampler that genuinely narrows the output distribution is doing something
none of these post-hoc corrections can.

### 2026-11-24 — Screening sweep: six variants, short runs, paired evaluation

Running one full method at a time (6000 steps plus a 16-seed evaluation, ~3.3 h each) tests about
three ideas a night. The sweep below tests six in roughly the same budget, by shortening the
training and — more importantly — cutting the evaluation to 8 seeds.

**Why 8 seeds is sound here, when 16 was required before.** The 16-seed requirement came from
*marginal* means: the baseline's own halves differ by 0.88 MAE. A **paired** comparison removes
that variance, because both runs see identical prompts, seeds and noise. Simulating a method that
genuinely shrinks each image's error by 25 %:

| seeds | effect | 95 % CI half-width | detected |
|---|---|---|---|
| 4 | −2.04 | 0.306 | yes |
| 8 | −2.04 | 0.246 | yes |
| 16 | −2.04 | 0.183 | yes |

Even four seeds separates it. Eight is used, against a matching 8-seed slice of the frozen
baseline. Full 16-seed numbers are reserved for whatever wins.

**The variants, each targeting something the analysis identified:**

| run | change | reason |
|---|---|---|
| `d_screen` | Method D | the only count term that is symmetric in over/under-counting *and* drops when objects merge |
| `a_screen` | Method A, `tau_g` 0.2 → 0.45 | `L_gap` previously read 0 across its whole useful range |
| `c_screen` | Method C, `tau_q` 4.0 → 1.0, `r_s` 1.25 → 1.5 | the k-means kernel was wider than the separation it had to enforce |
| `c_balance` | Method C, `act` ×4, `supp` ×0.25 | `L_supp` was 3.60 of a 3.87 total and punishes extra peaks far harder than missing ones — the asymmetry that aimed C at collapse |
| `b_screen` | Method B | the only error signal grounded in the decoded image rather than attention geometry |
| `d_delta` | Method D, `delta` 0.3 → 0.15 | δ=0.3 let the latent drift far enough to lose colour fidelity |

**A scheduling fact that unblocked this.** `PromptSampler.sample` picks a template at random per
batch, so training is template-agnostic and the prompt-template question is purely a measurement
choice, settleable afterwards on any checkpoint. It does not gate method training, so the sweep
runs first and the template comparison becomes an evaluation-only addendum.

**What counts as success**, per the ceiling analysis: not a lower MAE on its own. The baseline
already has slope +1.03, and perfect centring without variance reduction scores −1.18. The
quantities to read are the **per-n spread** and `seed_spearman_mean`, with MAE as a secondary.

### 2026-11-24 — Method D pre-flight on real attention maps: the layer and the threshold were both wrong

Method D was calibrated on planted Gaussians. Before spending a GPU slot on it, the same loss was
run on **real** cross-attention maps from the frozen generator (CPU, no contention with the
running evaluation). Three things came out, in order.

**1. The surrogate is faithful; the threshold was not.** `chi_soft` tracks `chi_hard` on real maps
at correlation **0.995**, mean absolute difference 0.26 — the differentiable relaxation is sound.
But at `tau_frac` = 0.5 (the planted-Gaussian value) a real map is 77–80 % foreground, so it
collapses to one blob and χ = 1 whatever n is:

| prompt | n=2 | n=4 | n=8 | occupancy |
|---|---|---|---|---|
| red circles | 1.0 | 1.0 | 1.0 | 0.77–0.80 |
| green crosses | 1.0 | 1.0 | 1.0 | 0.44–0.55 |
| blue diamonds | 1.3 | 2.4 | 6.1 | 0.18–0.27 |

Real attention maps are far less peaked than planted Gaussians. Calibrating a relaxation on
synthetic inputs and shipping it is the same class of error as the config bugs earlier in this log.

**2. A near-miss: "attention does not carry the count".** Sweeping the threshold on real maps, the
best 16×16 setting gave χ = 1.2, 2.2, 3.0, 2.5, 2.9 for n = 1, 2, 4, 6, 8 — saturating at about
three components. Repeating across 16×16, 32×32 and 64×64 layers, no resolution tracked n
(correlations +0.26 to +0.49; at 64×64, n=1 fragmented into 14 components). The conclusion drawn
was that the object cross-attention map simply does not encode count as separable regions, which
would condemn Methods A, C and D together.

**3. The control that overturned it.** All of those images were generated from
`data.templates[0]`, the bare noun phrase that produces edge-to-edge tiling. Repeating with
`data.templates[2]`, the countable phrasing, on identical seeds and prompts:

| layer, threshold | n=1 | 2 | 3 | 4 | 6 | 8 | corr(χ, n) |
|---|---|---|---|---|---|---|---|
| 16×16, τ=0.70, tiling prompts | 1.6 | 2.1 | 2.1 | 1.6 | 2.6 | 2.9 | +0.839 |
| 32×32, τ=0.70, **tiling prompts** | 4.3 | 3.0 | 3.5 | 5.2 | 3.5 | 5.1 | **+0.422** |
| 32×32, τ=0.70, **countable prompts** | 2.5 | 4.3 | 4.2 | 5.4 | 7.2 | 9.2 | **+0.989** |

**The attention map does encode the count — when the image actually contains countable objects.**
At a 32×32 layer on countable scenes, χ tracks the requested count at r = +0.989. On tiling
prompts the same layer gives +0.42, because there is no count in the image to encode.

**Two configuration consequences**, both now in `configs/method_d.yaml`:

- `cross_layers` moves to a **32×32** layer (`up_blocks.2.attentions.0…attn2`), not the 16×16 one
  the pre-check chose. The pre-check scanned tiling images, where every layer is equally
  uninformative, so it was choosing between equally bad options.
- `tau_frac` 0.5 → **0.70**, calibrated on real maps rather than synthetic ones.

**And one training consequence, which may matter more than either.** `PromptSampler.sample` draws
uniformly from `data.templates`, so only about a third of training batches use the phrasing where
the count signal exists at all. The sweep's first run restricts `data.templates` to the countable
phrasing for **training as well as evaluation**, paired against a frozen baseline measured the same
way. If any configuration in this project is going to work, this is the one.

**Method, for the write-up.** This is the third time a conclusion reversed on a control: red
pentagon for the baseline response, blue pentagon for the template probe, and now tiling prompts
for the attention structure. Each time the first reading came from an unrepresentative slice. The
rule that keeps earning its place: **before concluding a signal is absent, check whether the input
was capable of carrying it.**

### 2026-11-24 — The countable template cuts the baseline spread by 58 %, which changes the verdict

Measured on CPU alongside the running evaluation: the frozen baseline, same seeds and prompts,
under the two phrasings. Four prompts × three seeds × n ∈ {1,2,3,4,6,8}, repaired counter.

| | spread | MAE | slope (n ≥ 4) | counted per n (1,2,3,4,6,8) |
|---|---|---|---|---|
| `templates[0]` (bare noun phrase) | 5.30 | 6.43 | +1.42 | 2.2, 3.9, 7.8, 12.1, 17.5, 17.8 |
| `templates[2]` (countable scenes) | **2.22** | 4.46 | +1.88 | 1.2, 4.6, 6.2, 9.0, 13.1, 16.5 |

**Spread falls 58 %.** The ceiling analysis established that spread, not bias, is what decides
whether anything here can beat a constant predictor. Carrying that through, with the
best-constant bar for n ∈ {1,2,3,4,6,8} at 2.000:

| | if a method centred the distribution perfectly | skill |
|---|---|---|
| `templates[0]`, spread 5.30 | MAE 4.128 | **−1.064** — still loses |
| `templates[2]`, spread 2.22 | MAE 1.840 | **+0.080** — wins |

**On countable scenes, removing the bias alone is enough for positive skill. On the tiling prompts
it is not, however good the method.** That is the sharpest statement of what the prompt template
was costing: not accuracy, but the *possibility* of success.

And removing bias is precisely what a count loss does — every method pushes the counted value
toward n. The thing none of them can do is reduce variance, and on countable scenes they no longer
have to.

**Both failures are still present on `templates[2]`** and should not be glossed over: the counted
mean is 8.43 against a requested mean of 4.0, so the generator still roughly doubles the count, and
the n ≥ 4 slope of +1.88 overshoots rather than tracking. The claim is narrow — the *residual*
after centring is now small enough to clear the bar, not that the task is solved.

**Caveat.** 72 images on CPU in float32 against 1440 on GPU in bfloat16 for the sweep's reference
run. The direction is large and consistent across all four prompts, but the magnitude is an
estimate until `eval_baseline_white_s8` lands. The simulated centring also assumes a roughly
symmetric residual, which the observed over-generation makes optimistic.

This is why `d_white` — Method D, 32×32 layer, τ=0.70, trained *and* scored on countable scenes —
runs first in the sweep. It is the only configuration where the analysis says a positive result is
reachable.

**Sweep revised on this result.** Every method variant now trains and evaluates on countable
scenes, since the analysis says tiling prompts cannot yield positive skill whatever the method
does. `d_screen` is kept on the default mixed templates as the control that isolates how much of
`d_white` is the template rather than the method. Final order:

| | run | what it tests |
|---|---|---|
| 1 | `d_white` | Method D, 32×32, τ=0.70, countable — every finding combined |
| 2 | `d_screen` | same method, mixed templates — the template control |
| 3 | `a_white` | Method A with `tau_g` fixed |
| 4 | `c_white` | Method C with `tau_q`/`r_s` fixed |
| 5 | `c_bal_white` | Method C with `act`×4, `supp`×0.25 |
| 6 | `b_white` | Method B, image-grounded |
| 7 | `b_screen` | Method B, mixed templates |
| 8 | `d_delta_white` | Method D with δ=0.15 |

Each pairs against a frozen baseline measured on **its own** task; comparing a countable-scene run
against a tiling baseline would be comparing different tasks and would read as a large spurious
win.

### 2026-11-24 — The three held-out shapes are three different tasks, and the evaluation averages them

Breaking the repaired baseline down by shape (all six colours, 16 seeds):

| shape | MAE | spread | slope (n ≥ 4) | counted mean | exact |
|---|---|---|---|---|---|
| cross | **4.79** | **4.57** | **+1.807** | 9.35 | 0.116 |
| diamond | 8.80 | 5.29 | +1.115 | 14.01 | 0.032 |
| pentagon | 10.92 | 8.54 | **+0.171** | 15.31 | 0.039 |

**Pentagon has essentially no count response** — slope +0.17, against +1.81 for cross. The
"saturation" this log spent the afternoon characterising is largely a pentagon effect; crosses and
diamonds track the request. SD-Turbo renders "pentagon" as a tiled polygon pattern rather than as
discrete objects (visible in `runs/figures/template_probe.png`, where blue pentagons tile under
every phrasing), so part of the evaluation is asking for something the generator cannot draw
discretely.

By colour the spread is narrower but still real: blue MAE 4.81 and counted mean 8.34, against
yellow 10.14 and 15.38.

**Consequence: the headline number averages a task that works with one that does not.** A method
that improves crosses would be masked by pentagons, and a method that only looks good on pentagons
is probably exploiting noise. Per-shape breakdowns should be reported alongside every method
result, which `scripts/compare_paired.py` already supports through its per-colour table and should
be extended to shape.

**Some (prompt, seed) rows already beat the constant bar.** Of the 288 (colour, shape, seed) rows,
**20 (7 %) have MAE below 2.5** without any sampler at all. The best is blue cross seed 3 at MAE
1.20, counting 2, 3, 7, 5, 5, 6, 7, 5, 9, 8 for n = 1…10 — genuinely tracking. The worst is green
pentagon seed 5 at MAE 23.70, counting 7, 37, 27, 33, 24, 30, 31, 32, 36, 35. Every one of the six
best rows is a cross; the three worst are pentagons.

This reframes the sampler's job in the most concrete terms yet: **the frozen generator already
counts well for 7 % of (prompt, seed) combinations. The task is to make the other 93 % behave like
those.** Δφ is a function of ε, so it observes exactly what distinguishes a good seed from a bad
one — though the earlier held-out test is a caution that fitting per-prompt corrections to this
structure overfits, so the usable version of this has to generalise across seeds, not memorise
them.

### 2026-11-24 — The counter was double-counting: nested boxes that IoU cannot suppress

Raised from inspection of `runs/figures/baseline_what_it_draws.png`: the blue-cross image at n = 4
visibly contains four crosses and the counter returned five. Drawing the detector's boxes
(`runs/figures/cross_detections.png`) shows why — one box spans a *pair* of crosses while others
sit on the individual ones, and all survive non-maximum suppression.

**IoU structurally cannot fix this.** For a small box of area a nested inside a large box of area
A, the IoU is a/A — smallest exactly when the nesting is most severe. OWLv2 routinely fires on both
a single object and on a group of them, so plain IoU at 0.5 keeps both and the count inflates. A
containment test (intersection over the *smaller* area) expresses what IoU cannot for boxes of very
different size.

`_nms` now suppresses a lower-scoring box when the overlap covers ≥ 80 % of its own area, alongside
the existing IoU test. On the blue-cross row:

| asked | old NMS | with containment |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 3 | **2** |
| 3 | 4 | **3** |
| 4 | 6 | **4** |
| 5 | 6 | **5** |
| 6 | 6 | 5 |

Exact for n = 1…5, against being wrong on every count but n = 1 before.

**The rule must not merge real neighbours**, since the synthetic scenes place objects touching on
purpose. Pinned in `tests/test_hard_counters.py`: well-separated, touching and half-overlapping
pairs all survive; only the nested pair is suppressed.

**This is the fourth measurement defect found today**, after the proper-noun query, the threshold
set three times the library default, and the padded-coordinate bug — and the first found by
looking at the pictures rather than at the numbers. The evaluation audit predicted it
("concentric duplicates survive unless their widths are within 1.41×") and recommended exactly
this fix; it was deferred as "not the main lever", which was wrong.

**Consequence.** `eval_baseline_fix` and Method C's evaluation were both computed with the old
rule and will need re-measuring. Before spending GPU on that, the effect is being quantified
across all three shapes on CPU, because a fix validated on the one case that motivated it is not
evidence.

### 2026-11-24 — Quantified: the containment fix, and the frozen model beating the bar on crosses

144 images (4 colours × 3 shapes × 2 seeds × n ∈ {1,2,3,4,6,8}), same detector and threshold, only
the suppression rule changed. Best-constant bar for this count set is 2.000.

| shape | old MAE | **new MAE** | old exact | **new exact** | counted mean | **new skill** |
|---|---|---|---|---|---|---|
| all | 4.10 | **2.95** | 0.11 | **0.29** | 7.67 → 5.15 | −0.475 |
| **cross** | 3.23 | **1.81** | 0.15 | **0.33** | 7.06 → 4.85 | **+0.095** |
| diamond | 4.02 | **2.92** | 0.10 | **0.33** | 8.02 → 5.79 | −0.460 |
| pentagon | 5.06 | **4.12** | 0.08 | **0.21** | 7.94 → 4.79 | −1.060 |

**Exact accuracy nearly triples** (0.11 → 0.29) and MAE falls 28 % with no change to the
generator, the threshold or the query — only to how overlapping boxes are suppressed.

**On crosses, the frozen baseline now beats the constant predictor: skill +0.095.** That is the
first positive skill recorded in this project, and it comes from the *unmodified* generator. The
sampler is not needed to clear the bar on that shape; it is needed to extend it to the others.

**Much of the "over-generation" was the detector, not the generator.** The counted-to-requested
ratio falls from 1.92× to 1.29× overall, and from 1.99× to 1.20× on pentagons. This log has
described a generator that "counts, then adds a large surplus" and quantified that surplus at
+8.5 objects; a substantial part of it was one box drawn around a group of objects while others
were drawn around its members.

**What this invalidates, again.** `eval_baseline_owl`, `eval_baseline_fix` and Method C's
evaluation were all computed with the old suppression rule. Every absolute figure derived from
them — the +8.5 offset, the spread of 7.15, the ceiling analysis thresholds — is inflated by
double-counting and needs recomputing. The *relative* comparisons within each protocol remain
valid, since both sides were measured the same way.

**The sweep inherits the fix**, since its baselines and variants all run after it. Each sweep
comparison is therefore internally consistent. Method C's evaluation, still running against
`eval_baseline_fix`, is consistent with *that* baseline but not with the sweep's — a caveat to
carry rather than a result to discard.

**Pattern, now with four instances.** Proper-noun queries, a threshold three times the library
default, padded coordinates, and nested boxes. Every one inflated or distorted the headline, and
three of the four were predicted by the evaluation audit before being found. The conclusion for the
write-up is not "the counter was bad" but something sharper: **on a task measured by a learned
detector, the detector is a component of the system and needs the same validation as the model.**

**Re-measuring the baseline without re-generating it.** The evaluation saves every image, and the
generator is deterministic given a seed, so a *counter* change can be applied to the 2880 existing
grid cells directly — no GPU, no regeneration. `scratchpad/recount.py` does this on CPU alongside
the running GPU work and writes `runs/eval_baseline_nms/` in the same schema, so the existing
analysis tools read it unchanged. It costs CPU time that was otherwise idle and frees the GPU
entirely for the sweep.

This is worth remembering as a general property of the setup: **any change to how images are
scored can be re-measured for free; only changes to how they are generated need the GPU.** Three
of the four measurement defects found today (query, threshold, suppression) are scoring changes.

### 2026-11-24 — Method C "wins" on MAE and is a total failure

`c_tuned_s0` evaluated against `eval_baseline_fix` — same seeds, same prompts, same counter, 2880
images each.

```
MAE  8.166 -> 5.311      better by 2.855
paired verdict: candidate better, 95% CI [-3.64, -2.07]
improved on 58.2% of images, worse on 31.7%
```

By mean absolute error, by skill (−2.266 → −1.125) and by a seed-bootstrapped paired test on
identical images, Method C is a clear improvement. It is not:

| | baseline | Method C |
|---|---|---|
| counted mean | 12.89 | **0.19** |
| zero-detection rate | 1 % | **82 %** |
| slope (n ≥ 4) | +1.031 | **+0.023** |
| `seed_spearman_mean` | +0.667 | **−0.421** |
| spread | 7.15 | 0.29 |

**It emits almost nothing.** 82 % of its images contain no object the counter can find, and the
mean count is 0.19 against a requested mean of 5.50. The frozen generator *over*-counts, so a
sampler that produces nothing lands closer to n on average than one that produces too much. The
spread collapse of −96 % is not consistency, it is emptiness.

The pre-registered criteria called this exactly: *"degenerate: MAE improves while the slope goes
flatter"*. The slope went from +1.031 to +0.023 — the count is no longer followed at all — and the
seed correlation went **negative**.

**This is the justification for everything built around the headline number today.** Reported as
MAE plus a significance test — the obvious way — Method C is a publishable improvement. Reported
with the response slope, the counted mean and the zero rate, it is obviously broken. The same run,
the same data, opposite conclusions.

`scripts/compare_paired.py` now refuses to let that pass. It flags a degenerate win when the
candidate's zero rate exceeds 40 %, its counted mean falls below half the requested mean, or its
response slope drops under +0.2, and prints what the two runs actually emitted:

```
  verdict: **candidate better**

  *** DEGENERATE: a lower error here does not mean a better model ***
    - 82% of images contain nothing the counter can find
    - counted mean 0.19 against a requested mean of 5.50
    - response slope +0.023 - the count is no longer followed
    reference emitted 12.89 objects per image, candidate 0.19
```

The guard stays silent on a healthy comparison. It will now run automatically on every sweep
variant.

**Caveat.** Both runs here were scored with the old suppression rule, so the absolute figures are
inflated by the double-counting found later. The comparison between them is unaffected — both
sides measured identically — and the degeneracy is not a counter artefact: 82 % empty images is
not something a suppression rule invents.

**Operational note: CPU work is not free alongside GPU work.** The "re-count on CPU for free"
claim above needs qualifying. With the re-count running, the GPU evaluation slowed from 5.4 to
**13 minutes per grid** — a 2.4× slowdown — because the evaluation's own CPU-side work (VAE decode
scheduling, image encoding, data movement) was competing for cores against a process using 16
torch threads at 93 % total load. The GPU was not the bottleneck; feeding it was.

Dropping the re-count to `BelowNormal` priority and raising the evaluation to `AboveNormal`
restores the ordering without losing the 57 minutes already spent. The general point for this
machine: **an 8 GB GPU pipeline is partly CPU-bound, so "idle CPU" is not actually free capacity**
— it should be used at lower priority, or the GPU work pays for it.

### 2026-11-24 — THE BASELINE, corrected. Supersedes every earlier baseline number in this log

`runs/eval_baseline_nms`: the same 2880 images as `eval_baseline_fix`, re-counted on CPU with all
three counter repairs (threshold 0.1, colour out of the query, containment suppression). Nothing
about the generator changed.

| | original protocol | intermediate | **corrected** |
|---|---|---|---|
| | thr 0.3, colour query, IoU NMS | thr 0.1, shape query, IoU NMS | **+ containment** |
| MAE | 4.880 | 8.166 | **5.210** |
| skill | −0.952 | −2.266 | **−1.084** |
| exact | 0.1181 | 0.0622 | **0.1455** |
| slope (n ≥ 4) | +0.273 | +1.031 | **+0.595** |
| spread | 5.67 | 7.15 | **6.01** |
| counted mean | 6.51 | 12.89 | **7.46** |
| zero rate | 17.5 % | 1.0 % | **1.6 %** |

**Both of this log's earlier headline claims about the slope were wrong, in opposite directions.**
The original protocol's +0.273 said the generator saturates; that was missed detections flattening
the curve. The intermediate +1.031 said it tracks the count almost perfectly; that was duplicate
boxes amplifying the response, since more objects produce more nested detections. The truth is
**+0.595** — partial tracking with partial saturation — and the fitted relationship for n ≥ 4 is

```
counted = 0.595 · n + 4.96
```

not the `1.031·n + 8.51` recorded earlier. The surplus is about 5 objects, not 8.5.

**Per shape, and the shapes could hardly be more different:**

| shape | MAE | spread | slope (n ≥ 4) | counted mean | exact |
|---|---|---|---|---|---|
| cross | **3.89** | **4.29** | **+1.906** | 7.54 | **0.184** |
| diamond | 5.82 | 5.77 | +0.523 | 8.89 | 0.150 |
| pentagon | 5.92 | 6.17 | **−0.645** | 5.95 | 0.102 |

**Pentagon's response slope is negative.** Asking for more pentagons produces *fewer* detected
objects. Cross overshoots at +1.906, diamond under-tracks at +0.523. Averaging a +1.9, a +0.5 and
a −0.6 into a single +0.595 describes none of them. Per-shape reporting is not a refinement here,
it is required for the number to mean anything.

**Ceiling, recomputed.** A method that centres the distribution at each n and shrinks the spread by
a factor s:

| s | MAE | skill |
|---|---|---|
| 1.00 — centring only | 4.991 | −0.996 |
| 0.70 | 3.494 | −0.397 |
| **0.50** | 2.496 | **+0.002** |
| 0.40 | 1.996 | +0.201 |

The bar is **halve the spread**, from 6.01 to about 3.0 — easier than the 40 % the inflated data
implied, and still demanding. Centring alone remains insufficient, which was the one conclusion
that survived all three protocols.

**What this supersedes.** Every absolute baseline figure earlier in this log — the +8.5 surplus,
the spread of 7.15, the 40 % threshold, the "slope is already 1.0, the methods need only remove
the bias" framing — is replaced by the table above. The *relative* comparisons (Method C against
`eval_baseline_fix`) remain valid because both sides used one protocol, but Method C must be
re-scored against this baseline before it goes in any final table.

### 2026-11-24 — Method C's "win" was the inflated baseline

Placing Method C beside the corrected baseline, both on the same 2880 images:

| | MAE | skill | slope (n ≥ 4) | counted mean | zero rate |
|---|---|---|---|---|---|
| `eval_baseline_nms` (corrected) | **5.210** | −1.084 | +0.595 | 7.46 | 1.6 % |
| `c_tuned_s0/eval_fix` (old suppression) | 5.311 | −1.125 | +0.023 | 0.19 | 82 % |

**Method C is worse than doing nothing.** Its reported advantage of 2.855 MAE, significant on a
seed-bootstrapped paired test over identical images, existed only because the baseline it was
measured against was inflated by double-counting. Against a correctly counted baseline the
advantage is −0.10 — a small loss.

The two runs still used different suppression rules, so this is not yet a like-for-like number.
But the direction is safe to state in advance: the containment fix removes *duplicate* detections,
and 82 % of Method C's images contain nothing to duplicate, so its counts can barely move while
the baseline's fell from 12.89 to 7.46. The re-count is running to confirm it rather than leave it
as an inference.

**Two lessons, both worth the write-up.**

A degenerate solution can beat a *broken* measurement by a significant margin on the correct
statistical test. The paired bootstrap was right about what it measured — Method C really is
closer to n than the over-counting baseline was — and the conclusion was still wrong, because the
reference was wrong. Significance testing protects against noise, not against a mis-specified
comparison.

And the degeneracy guard earns its place twice over: it flagged this run on the images'
*content* (82 % empty, counted mean 0.19, slope +0.023) at a point when every error statistic
said the method had worked.

**Operational: background work survives a session restart, monitoring does not.** The Claude
session ended and reported all 19 background tasks stopped. In fact every GPU and CPU job was
still running — the evaluation, the sweep's shell, the re-count — because they were orphaned
rather than killed. What died was the *task tracking* and the result watchers. The recovery is to
re-attach by inspecting `runs/` and the process list, not to restart the work: relaunching the
sweep would have discarded a 13/18 evaluation and started a second GPU job alongside the first.

Worth knowing for a long overnight run: **check what is actually running before assuming a
reported stop is real.**

### 2026-11-24 — Method D, first 300 steps: the over-count is falling

`d_white` — Method D, 32×32 layer, τ=0.70, trained and scored on countable scenes — logged in
100-step blocks:

| block | `D/chi_hard_minus_n` | `D/chi_hard` | soft/hard ratio |
|---|---|---|---|
| 0–100 | **+2.76** | 8.39 | 0.54 |
| 100–200 | **+1.73** | 7.22 | 0.58 |
| 200–300 | **−0.11** | 5.50 | **0.69** |

First-half against second-half: −2.32, standard error 0.81, **t = −2.86**. That is not noise.
Two things are moving in the right direction at once: the discrete component count is approaching
the requested count, and the differentiable surrogate is converging on the discrete one it stands
in for (0.54 → 0.69), because training sharpens the attention map and a sharper map is where the
relaxation is faithful.

**Mechanically clean.** Zero skipped steps, 1.2 s/step, 6.55 GB stable. `loss/area` is usually 0,
so objects are above the size floor, and `loss/bin` sits near 0.002, so the occupancy field stays
near-binary — the uniform-haze degeneracy the design analysis predicted is not occurring.

**What this is not.** This is `chi_hard` on the *attention map*, the method's own diagnostic. Method
C improved its own objective by 31× while producing one object per image and then none. A count of
attention components is not a count of objects: **n tally marks would satisfy it exactly as well as
n objects**, which is the limitation declared when Method D was written. The evaluation with OWLv2
on the decoded images is the test that matters, and the previews are the first visual check.

**The specific risk now.** At step 290 the over-count is −0.11, essentially on target. The fitted
trend is −0.016 per step, which extrapolated over 3000 steps would reach −48 — absurd, so it must
flatten or reverse. If it does not flatten, Method D overshoots into *under*-counting, which is the
same destination Method C reached by a different route. Watching whether it settles near zero is
now the live question, and the degeneracy guard will catch the endpoint either way.

### 2026-11-24 — Correction: Method D's "clear trend" was a dip in a truncated series

The previous entry reported the over-count falling monotonically across 100-step blocks, with
t = −2.86 on a first-half/second-half comparison, and called it "not noise". That test was run on
the series truncated at step 290. Extending it:

| block | `chi_hard − n` | `geom/d_anchor_mean` | `geom/d_pair_mean` |
|---|---|---|---|
| 0–100 | +2.76 | 0.026 | 0.0007 |
| 100–200 | +1.73 | 0.083 | 0.0014 |
| 200–300 | **−0.11** | 0.089 | 0.0024 |
| 300–400 | **+1.90** | 0.082 | 0.0016 |
| 400–600 | **+2.86** | 0.045 | 0.0010 |

The over-count returned to its starting value. **The significance test was correct about the data
it was given and the data was half the story** — the window happened to end at the bottom of a
dip. Running a two-sample test on a time series truncated at the current step is a way of
discovering that the series went down recently, not that it has a trend. This log already carries
the same error in another form: a positive skill on crosses computed from a 144-image sample that
dropped the hardest counts.

**Where Method D actually stands at step 600.** The sampler has moved barely at all:
`d_anchor_mean` 0.045 against a δ budget of 0.30, having peaked at 0.089 and come *back*, and
`d_pair_mean` 0.001, meaning the latent is essentially identical across counts. The step-500
previews (`runs/figures/d_white_step500.png`) match the frozen baseline closely, which is exactly
what a near-identity sampler should produce.

**This is still a meaningful difference from Method C.** At step 500 Method C had driven
`d_anchor` to 0.29, destroyed the images into washed-out fields with blue borders, and begun
shifting blue pentagons toward red. Method D at the same step has intact, countable images with
correct colours. The failure mode being avoided is real; what is missing is any evidence of the
method *doing* something.

**Two readings, and the run will separate them.** Either the objective is too weak to move the
sampler within 3000 steps — `loss/area` is usually 0 and `loss/bin` ~0.002, so `L_cc` is carrying
the whole gradient against `anc` and `pri` at weight 10 — or it is correctly finding that this
generator cannot do better and is staying near the identity it was initialised at. The weights are
the first thing to vary if the former.

### 2026-11-24 — Why Method D barely moves: the gradient is large, noisy, and clipped to nothing

The obvious explanation for a near-identity sampler is a weak loss. Measured over the first 600
steps, against Method C which moved a great deal:

| | `loss/total` | `grad_norm` | `d_anchor_mean` |
|---|---|---|---|
| Method D (`d_white`) | 0.508 | **141.3** | 0.065 |
| Method C (`c_tuned_s0`) | 1.865 | **33.7** | 0.210 |

**Method D's gradient is four times larger than Method C's and it moves three times less.** The
loss is not too weak. Two things are:

**`grad_clip` is 1.0 and the gradient norm is two orders of magnitude above it.** Method D's
grad_norm runs min 16.8, median 112.7, max 594.3, and **100 % of steps are clipped** — by a median
factor of 113. Only the update *direction* survives; magnitude is discarded on every step of every
run. This is not specific to Method D: Method C's 33.7 is clipped by 34×. The configured value was
never revisited against a measured gradient.

**The direction is inconsistent.** Method D's gradient norm varies 35× between steps because χ is
estimated from the 8 samples in a micro-batch (`eps_per_batch` 1 × `counts_per_eps` 4 ×
`grad_accum` 2) and `D/chi_hard` bounces between 3.6 and 12 across consecutive logged steps. With
magnitude clipped away, inconsistent directions cancel and the latent stays where it started —
which is exactly what `d_anchor` peaking at 0.089 and returning to 0.045 looks like.

Method C's gradient pointed the same way every step — *suppress everything* — so its updates
accumulated and it reached the δ boundary by step 500. A consistent wrong direction moves further
than an inconsistent right one.

**Two candidate fixes, neither yet run:**

1. **Raise `grad_clip`.** At 1.0 it is binding on every step of every method in this project. The
   value should be set from the measured distribution, not left at a default that silently
   discards 99 % of the signal.
2. **Reduce the gradient noise.** More samples per step via `grad_accum`, at a proportional cost
   in wall time — `grad_accum` 2 → 4 halves the step rate and reduces the noise by √2.

The first is free and the more likely culprit. It is also a project-wide finding: every result in
this log was produced under a clip that binds on every step.

### 2026-11-25 — grad_clip set from the measured distribution (1.0 → 200)

**Correction to the previous entry first.** It quoted Method C's gradient norm as 33.7 against
Method D's 141.3. Those are *means*, inflated by outliers. The medians are **2.2 for Method C and
93.5 for Method D** — so at `grad_clip` 1.0 the clip reduces Method C's typical step by 2.2× and
Method D's by 93×, a 40× asymmetry rather than the 4× implied. The conclusion holds and is
stronger than stated.

Percentiles over the first 860 steps:

| run | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| Method D | 40.7 | 64.8 | **93.5** | 179.2 | 274.3 | 594.3 |
| Method C | 1.2 | 1.5 | **2.2** | 4.6 | 20.8 | 165.3 |

**Choosing the value from the data.** A clip should tame outliers, not the typical step. For
Method D:

| clip | steps clipped | median reduction |
|---|---|---|
| 1.0 (default) | 100 % | 93.5× |
| 50 | 81 % | 2.5× |
| 100 | 48 % | 1.8× |
| **200** | **16.5 %** | **1.7×** |
| 300 | 9 % | 1.4× |

**200** is the rerun's value: the typical step (93.5) passes untouched, p90 (274) is trimmed 1.4×
and the worst (594) 3×.

`d_white` at clip 1.0 is preserved as **`d_white_clip1`** (860 steps) so the two are comparable at
matched steps — the only change between them is the clip.

**What would count as the clip being the cause.** At clip 1.0, `d_anchor_mean` peaked at 0.089
against a δ budget of 0.30 and fell back to 0.045, and `d_pair_mean` stayed at 0.001, meaning the
latent barely differed between counts. If the clip was what prevented the updates accumulating,
`d_anchor` should now climb and keep climbing. If it stalls at the same place, the problem is the
objective or the generator, not the optimiser, and raising the clip was the wrong lever.

**Operational: `sh.exe` is not `bash.exe`, and orphaned chains keep going.** Clearing processes
before the rerun took four attempts because every filter was written as
`Name -eq 'bash.exe'`, while the chain shells were running as **`sh.exe`** — the scripts are
launched with `bash` but re-exec through `sh`, so the filter silently matched nothing and the
shells survived each sweep. Two consequences followed:

- A surviving shell restarted the default-template baseline at 23:58:44, one minute before the
  rerun, and the two shared the GPU. The rerun produced no log line for nine minutes and looked
  hung; it was simply getting half a card.
- Killing that baseline did not stop the shell, which moved on to its *next* variant (`d_screen`)
  and started a second training job, which then outlived its parent as an orphan.

The reliable procedure is to match on the **command line** rather than the process name, trace
`ParentProcessId` to find what is spawning the work, and kill the parent before the child —
otherwise the parent simply starts the next stage. Confirm with `nvidia-smi` memory rather than a
process count: 7640 MiB with "one" job running is the signal that something else is on the card.

### 2026-11-25 — grad_clip 200: the sampler moves, and keeps moving

Matched at step 860, the only difference between the runs being the clip:

| run | `chi_hard − n` | `d_anchor_mean` | `d_pair_mean` | `grad_norm` |
|---|---|---|---|---|
| `d_white_clip1` | +1.83 | 0.0700 | 0.0012 | 137.5 |
| `d_white_clip200` | +1.40 | 0.0822 | 0.0013 | 125.1 |

The aggregate difference is small. The *trajectory* is not:

| block | clip 1.0 `d_anchor` | clip 200 `d_anchor` |
|---|---|---|
| 0–200 | 0.026 | 0.035 |
| 200–400 | 0.086 | 0.056 |
| 400–600 | 0.082 | 0.071 |
| 600–900 | 0.045 | **0.146** |

**At clip 1.0 the sampler peaked at 0.089 and came back to 0.045. At clip 200 it is at 0.146 and
still rising** — over three times the old endpoint, against a δ budget of 0.30. The prediction
written before the rerun was that if clipping was the constraint, `d_anchor` would climb and keep
climbing; so far it does.

`chi_hard − n` tracks it down: +2.81 → +2.65 → +1.42 → −0.58.

**Stated as pending, not established.** The 600–900 block is the last one, and this log already
contains one case where a monotone-looking run through the final block reversed as soon as more
data arrived — the earlier "t = −2.86" on a series truncated at step 290. The difference this time
is that clip 1.0's own trajectory provides the control: it had already turned back by its fourth
block, and clip 200 has not. Confirmation needs blocks past step 1200, which is running.

**What has not changed.** `d_pair_mean` is 0.0013, essentially identical to clip 1.0's 0.0012. The
sampler is moving further from ε but moving *every count the same way*. A perturbation identical
across counts cannot change the count, whatever it does to the image — the same observation made
of Method C at step 450, where `d_pair/d_anchor` was 0.15. Here it is 0.009. Raising the clip has
let the sampler move; it has not yet made the movement depend on n.

### 2026-11-25 — clip 200 through step 1900: a damped oscillation, and a correction

**Correction first.** The previous entry said that at clip 200 `d_anchor` was "still rising" and
predicted it would keep climbing. It did not. Block means:

| block | `chi_hard − n` | `d_anchor` | `d_pair` |
|---|---|---|---|
| 0–200 | +2.81 | 0.035 | 0.0014 |
| 200–400 | +2.65 | 0.056 | 0.0017 |
| 400–600 | +1.42 | 0.071 | 0.0007 |
| 600–900 | −0.50 | 0.143 | 0.0015 |
| 900–1200 | **−1.74** | **0.171** (peak) | 0.0040 |
| 1200–1500 | −0.75 | 0.138 | 0.0010 |
| 1500–1900 | **+0.47** | 0.116 | 0.0005 |

`d_anchor` peaked at 0.171 and fell back to 0.116 — the same rise-and-return shape as clip 1.0,
at roughly twice the displacement (clip 1.0 peaked at 0.089, settled at 0.045). The clip raised
the ceiling; it did not remove it. **Three times tonight a trend has been called from the final
block of a series and three times the next block changed it.** The rule that keeps being
re-learned: with a noisy diagnostic, the last block is the least trustworthy point in the series.

**What is genuinely new, and is not a drift.** The count error crosses zero twice and its
amplitude halves: mean |error| over the first three blocks is 2.29, over the last three 0.99.
That is a **damped oscillation about the target**, which is what a converging controller looks
like, as against Method C's monotone collapse to −n. On its own diagnostic, Method D is settling
on the requested count rather than running away from it.

**The caveat the evaluation will settle.** `D/chi_hard_minus_n` is a *batch mean over mixed n*,
and `d_pair_mean` is 0.0005–0.004, meaning the perturbation is nearly identical across counts. A
count-independent perturbation can drive that mean to zero by under-counting small n and
over-counting large n — the mean would read 0 while every individual prompt is wrong. **Mean zero
does not imply per-n zero**, and only the per-count evaluation distinguishes them. This is the
same shape of error as reading a good MAE off Method C while it emitted 0.19 objects per image.

So the honest position at step 1900: the optimiser constraint was real and raising the clip
removed it, the count term is converging on its own diagnostic, and whether any of that reaches
the *images* is unresolved until the OWLv2 evaluation runs.

### 2026-11-25 — L_ord has never been active in any Method D run

**The run ended at step 2500, not 3000.** The session that launched it exited and this time the
orphaned chain did not survive: `log.jsonl` stopped at 10:05 and the GPU sat idle for 5.5 hours.
`sampler_step002000.pt` is the newest checkpoint; there is no `sampler_last.pt`, so the chain's
evaluation never started. Previous orphan scares were false alarms — processes outlived the
session — so this one was checked by file mtime against the clock rather than by a process
listing, which is the only check that distinguishes "still running" from "stopped an hour ago".

**The full block table, through 2500:**

| block | `chi_hard − n` | `d_anchor` | `d_pair` | `grad_norm` | `loss/ord` |
|---|---|---|---|---|---|
| 0–300 | +2.91 | 0.043 | 0.00158 | 157.9 | 0.451 |
| 300–600 | +1.67 | 0.065 | 0.00094 | 105.2 | 0.422 |
| 600–1000 | −0.74 | 0.152 | 0.00226 | 122.6 | 0.408 |
| 1000–1500 | −1.21 | 0.150 | 0.00208 | 77.7 | 0.432 |
| 1500–2000 | +0.46 | 0.116 | 0.00042 | 95.9 | 0.422 |
| 2000–2300 | −1.04 | 0.158 | 0.00050 | 49.3 | 0.440 |
| 2300–2500 | −0.30 | 0.128 | 0.00027 | 96.0 | 0.454 |

The damped oscillation held up over 600 further steps, which is worth recording because the last
three calls made from a final block did not: the error crosses zero three times and mean |error|
falls from 1.77 over the first three blocks to 0.60 over the last three. That is a real property
of the run, not a trend read off its last point.

**`d_pair` went the wrong way the whole time: 0.00158 → 0.00027.** The sampler displaces the noise
by `d_anchor` ≈ 0.13 and displaces it *more and more identically* across counts. By step 2500 the
perturbations for n = 1 and n = 10 differ by 0.00027 in normalised distance while each sits 0.128
away from ε — a ratio of 1 : 474. Whatever that perturbation does to the image, it does the same
thing whichever count was asked for.

**Why: the one term that forbids this has never run.** `L_ord = Σ_{n<m} (d(z_n, z_m) − α g(n,m))²`
with α = 2δ/log n_max = 0.2606 targets `d_pair` = 0.0475 for adjacent counts and 0.600 for (1, 10).
Against the measured 0.00027, it is the only term in the entire objective that requires the
perturbation to *differ* per count — `L_cc`, `L_bin` and `L_area` are each computed per sample and
are all satisfied by one count-independent displacement that lowers the average χ error, which is
exactly the solution found. `L_anc` and `L_pri` only constrain magnitude.

`configs/method_d.yaml` stages `ord` to start at step 3000, and every Method D run has been 3000
steps. **It has never once been active.** `loss/ord` is logged every step — 0.41–0.45, flat
across the whole run — because the method computes it and then drops it from the weighted sum.
Reading that column as if it were part of the loss is what hid this for three runs: the number was
on screen the entire time, at a value 1000× its target, and it was being reported, not minimised.
Method B carries the identical staging (`ord` at 3000) and so has the same latent defect.

Spec Remark 1 says L_ord alone is satisfied by a collinear embedding and "only adds smoothness".
That is true and is the point: it cannot supply the count signal, but it is what stops the count
signal from being answered with a constant. `L_cc` picks *which* displacement lowers the χ error;
`L_ord` forces that displacement to depend on n.

**`configs/method_d_ord.yaml`** is `method_d.yaml` with a single change — one stage, from step 0,
including `ord`. Verified on a 30-step smoke run: `loss/total` 0.849 against the ~0.45 the same
step would have shown before, and `d_pair` rising 7.8e-9 → 4.6e-4 over 30 steps where clip 200 was
falling by then. That is the config now training as `d_ord`.

### 2026-11-25 — the anchor budget δ = 0.3 contains total collapse

**Look at the images.** `runs/figures/ord_vs_clip200.png`, blue pentagons, the same ε in every row:

| row | what it draws | mean luminance | within-image sd | col-to-col Δ |
|---|---|---|---|---|
| frozen baseline | pentagons on white, count roughly tracking | 157.1 | 46.6 | **29.8** |
| clip 200 @1000 | a solid dark-blue square | 52.9 | 59.4 | 6.0 |
| clip 200 @2000 | a solid **black** square | 34.5 | 78.8 | 3.4 |
| d_ord @500 | black, a few pentagons at the edges | 14.4 | 43.4 | 5.5 |
| d_ord @1000 | solid **red** for n ≤ 6; ten real pentagons at n = 10 | 90.5 | 17.7 | 10.6 |

Both Method D variants have destroyed the generator. Not degraded — replaced the scene with a
single flat colour. And they did it *inside* the anchor budget: `d_anchor` was 0.13–0.22 against
δ = 0.3, so `L_anc` read exactly 0.0 for the whole run. The term meant to keep z_n in a safe
neighbourhood of ε never objected, because the neighbourhood it defines is not safe.

**The correction this forces.** The previous entry concluded: *"the optimiser constraint was real
and raising the clip removed it."* The mechanism was right and the implication was backwards.
`grad_clip: 1.0` was limiting how far the sampler could travel per step, and that limit was the
only thing holding earlier runs on the image manifold. Removing it did not free the method; it
let the method walk off the manifold. `d_white_clip1` held `d_anchor` at 0.045–0.089 and drew
objects. `d_white_clip200` reached 0.13–0.22 and draws a black square. **The collapse radius is
bracketed between 0.045 and 0.13, and δ = 0.3 is two to six times larger than the whole safe
region.** It is marked OPEN in base.yaml:50 and has never once been measured.

So the right reading of the grad-clip experiment is not "the clip was too tight" but "the clip was
doing the anchor's job, badly and by accident". The anchor should do it, with a δ that reflects
where the generator still draws scenes.

**What is nonetheless real in d_ord.** Turning `L_ord` on worked, on its own terms: `d_pair` went
0.0025 → 0.0101 → 0.105 over 900 steps, against 0.00027 at the end of clip 200 — a factor of 389 —
while `loss/ord` fell 0.629 → 0.254 → 0.217. And the count-dependence reaches the image: at
n = 10 the sampler draws ten pentagons where n ≤ 6 gives flat red. The ordering term does what it
was added to do. It was applied on top of a collapsed generator, so what it modulates is how much
of the frame is destroyed rather than how many objects appear.

Note also that the frozen baseline's col-to-col Δ of 29.8 is **three times** any sampler's. The
prompt alone already makes the image depend on n more than any trained sampler has. A method that
lowers that number is removing information the frozen model had.

**`delta_probe.py`** measures the radius directly: push ε a fixed normalised distance s in a
random direction, decode, and ask whether OWLv2 still finds objects. d(a,b) = ‖a−b‖/√D, so
z = ε + s·√D·u/‖u‖ lands exactly at s. Random directions give the generic radius; a *learned*
direction is chosen to change the image and can only be worse, so this is an upper bound on a
safe δ. Running now over s ∈ {0, 0.02, …, 0.30}.

### 2026-11-25 — correction: distance is not what breaks the generator

The entry above inferred a "collapse radius bracketed between 0.045 and 0.13" from the fact that
`d_white_clip1` held `d_anchor` at 0.045–0.089 and drew objects while `d_white_clip200` reached
0.13–0.22 and drew a black square. **That inference is wrong.** It assumed the difference between
the two runs was distance. `delta_probe.py` measured distance directly — push ε a fixed normalised
distance in a *random* direction and decode:

| d(z, ε) | mean count | zero % | corr(counted, n) | mean lum | img sd | verdict |
|---|---|---|---|---|---|---|
| 0.00 | 4.50 | 0.0 % | +0.698 | 171.8 | 51.9 | scene |
| 0.06 | 4.31 | 0.0 % | +0.691 | 171.7 | 51.9 | scene |
| 0.13 | 4.50 | 0.0 % | +0.708 | 171.3 | 51.9 | scene |
| 0.20 | 4.67 | 2.8 % | +0.728 | 170.2 | 52.3 | scene |
| **0.30** | 4.92 | 2.8 % | **+0.749** | 167.2 | 54.5 | **scene** |

At the full δ = 0.3 budget — a perturbation whose norm is 30 % of ‖ε‖ — the image is essentially
unchanged: luminance moves 171.8 → 167.2 and the count correlation does not degrade at all. A
random displacement of exactly the size that destroyed the generator when *learned* does nothing
when random.

**So the collapse is about direction, not distance,** and this is the ordinary adversarial-example
asymmetry: the one-step generator denoises isotropic perturbation back toward the data manifold,
while gradient descent finds the thin subspace that does not get denoised away. `d_white_clip200`
did not collapse because it travelled 0.13; it collapsed because it travelled 0.13 *in the
direction the loss pointed*.

**The consequence for the spec.** `L_anc` constrains ‖z_n − ε‖/√D, and that quantity has now been
measured not to control whether the output is a scene. Tightening δ would not have saved these
runs — it would only have made the sampler weaker while leaving the destructive direction
available at every radius. The anchor is nearly useless as written, and no choice of δ fixes it,
because the failure is not in the ball it defines.

The constraint has to be on the thing that actually degrades: the **output**. The natural
correction is an anchor in x̂0 space, `Σ_n max(0, d(x̂0(z_n), x̂0(ε)) − δ_x)²`, which says "change
the image, but not past the point where it stops being a scene" and is measured on the quantity
that collapsed. x̂0 is already produced by the forward pass, so this costs one extra U-Net
evaluation per ε (the unperturbed reference), not a VAE decode.

`direction_probe.py` is running the same radii with d_ord's *learned* direction to confirm the
asymmetry directly and to put a number on δ_x.

### 2026-11-25 — the count information is already there; the generator's gain is 1.5×

`eval_baseline_white_s8`: the frozen generator, countable-scene prompts, 1360 images over 17 of
the 18 (colour, shape) cells, counted with containment NMS at threshold 0.1 and no colour in the
query. Against the bare-noun baseline on the same counter:

| baseline | MAE | skill | spread | slope n ≥ 4 | counted mean | zero % |
|---|---|---|---|---|---|---|
| bare noun (`eval_baseline_nms`) | 5.210 | −1.084 | 6.01 | +0.595 | 7.46 | 1.6 |
| **countable scenes** | **4.632** | **−0.853** | **4.11** | **+1.550** | 9.29 | 0.7 |

Per shape, the template does something the aggregate hides — it **repairs the inverted response**:

| shape | slope, bare noun | slope, countable | spread, bare noun | spread, countable |
|---|---|---|---|---|
| cross | +1.906 | +2.351 | 4.29 | 3.26 |
| diamond | +0.523 | +1.405 | 5.77 | 2.92 |
| pentagon | **−0.645** | **+1.029** | 6.17 | 4.96 |

Pentagon counted *down* as the prompt asked for more. With countable prompts every shape counts up.

**The finding.** Fit `counted = a·n + b` on half the seeds and score on the other half:

```
counted = 1.514 n + 0.069
```

The offset is zero and the gain is 1.5. The generator is not failing to hear the number — it is
hearing it and drawing about half again too many. The per-count means make this plain:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| counted | **1.07** | 3.91 | 5.94 | 8.01 | 9.72 | 11.57 | 12.34 | 14.24 | 17.21 | 17.75 |
| ± sd | 0.40 | 3.75 | 2.90 | 3.72 | 4.13 | 3.71 | 4.60 | 5.63 | 6.83 | 8.05 |

Asked for one object it draws **1.07 ± 0.40**. The count is in there.

**Three reference points, all scored on held-out seeds, bar = 2.500 on the same rows:**

| | MAE | skill |
|---|---|---|
| countable baseline, raw | 4.632 | −0.853 |
| **integer prompt remapping** (ask for the n whose output is closest to the target) | **2.460** | **+0.016** |
| measurement-side affine correction (the ceiling) | 1.649 | **+0.341** |

Swapping the two folds gives +0.424 for the affine correction, so it is not an artefact of which
half was fitted. **+0.016 is the first positive skill this project has produced** by changing what
is generated rather than how it is scored, and it needs no training at all.

**Why this reframes the sampler's job.** Remapping is quantised: targets 1 and 2 both map to the
prompt "one", 4 and 5 both to "three", and the prompt cannot ask for 2.6 objects. That costs it
almost the whole gap between +0.016 and the +0.341 the same correction achieves when applied
continuously. **A perturbation of the initial noise is continuous**, and the gap between those two
numbers is exactly what it is for. The task is no longer "create count-sensitivity" — the frozen
model already has it at slope 1.5 — it is "correct a known gain", which is a far smaller ask and
one a bounded perturbation can plausibly meet.

It also says what to optimise. A term that reads the *image* and asks for n objects attacks the
gain directly; an attention statistic does not. That is Method B, and its counter is training now.

### 2026-11-25 — the learned direction is destructive at a 4× smaller radius than a random one

`direction_probe.py`, same ε and prompts, d_ord's step-1000 perturbation rescaled to each radius
against a random direction rescaled to the same radius:

| d(z, ε) | learned: count | zero % | img sd | random: count | zero % | img sd |
|---|---|---|---|---|---|---|
| 0.00 | 4.50 | 0.0 | 51.9 | 4.50 | 0.0 | 51.9 |
| 0.04 | 4.33 | 0.0 | 50.0 | 4.39 | 0.0 | 51.9 |
| 0.06 | 4.25 | 2.8 | 40.5 | 4.31 | 0.0 | 51.9 |
| 0.08 | 2.86 | 16.7 | 33.1 | 4.36 | 0.0 | 51.9 |
| 0.10 | 2.11 | 30.6 | 24.7 | 4.53 | 0.0 | 51.9 |
| 0.13 | 1.25 | **52.8** | 17.4 | 4.56 | 0.0 | 51.9 |
| 0.20 | 0.14 | **86.1** | 12.2 | 4.61 | 2.8 | 52.3 |
| 0.30 | 0.14 | **86.1** | 10.1 | 4.92 | 2.8 | 54.5 |

**Largest radius still drawing a scene: 0.08 learned, 0.30+ random.** Image sd decays monotonically
along the learned direction (51.9 → 10.1) and is flat along the random one (51.9 → 54.5). This is
the ordinary adversarial asymmetry — the one-step generator denoises isotropic perturbation back
toward the data manifold, and gradient descent finds the thin subspace that does not get denoised
away — and it settles the question the previous entry left open. **‖z − ε‖ is the wrong quantity
to constrain**, and no δ fixes `L_anc`: any δ large enough to let the sampler act is large enough
to let it destroy the image, because the two live at different radii in different directions.

It also explains both Method D runs exactly. `d_white_clip200` held `d_anchor` at 0.13–0.22 —
between 52.8 % and 86.1 % empty on this table — and drew black squares. `d_white_clip1` held
0.045–0.089, straddling the 0.06–0.08 edge, which is why its images were recognisable but poor.
`grad_clip: 1.0` was not a tuning accident; it was the only thing keeping that run inside the safe
radius, and raising it to 200 removed the last constraint that was doing any work.

**The harder half of the result.** At the largest radius where the learned direction still draws a
scene (0.06), the count has moved 4.50 → 4.25. The direction does not meaningfully change the
count *before* it destroys the image. For this direction — trained against χ on an attention map —
there is no usable operating point at all. That is a property of the direction, not of the
sampler form, and it is the strongest evidence yet that an attention statistic is the wrong signal.

### 2026-11-25 — prompt remapping, and where its error actually comes from

Scored on held-out seeds, bar 2.500:

| | MAE | skill | map (target 1…10 → prompt) |
|---|---|---|---|
| raw countable baseline | 4.632 | −0.853 | |
| global remapping | 2.460 | **+0.016** | 1123344556 |
| per shape: diamond | 1.613 | **+0.355** | 1123344455 |
| per shape: cross | 2.475 | +0.010 | 1233356677 |
| per shape: pentagon | 2.829 | −0.132 | 1112333556 |
| per-shape combined | 2.296 | **+0.082** | |
| affine ceiling (measurement-side) | 1.649 | +0.341 | |

Diamond alone reaches +0.355 — the full ceiling — so on a well-behaved shape the frozen model plus
a relabelling is already a working counter. Pentagon drags the average to +0.082.

**Decomposing the residual error of global remapping:**

| target | prompt used | mean counted | bias | sd | bias share of MAE |
|---|---|---|---|---|---|
| 1 | 1 | 1.07 | +0.07 | 0.40 | 100 % |
| 2 | 1 | 1.07 | **−0.93** | 0.40 | 94 % |
| 5 | 3 | 5.94 | +0.94 | 2.90 | 41 % |
| 10 | 6 | 11.57 | **+1.57** | 3.71 | 50 % |

The remaining error is mostly **bias, not spread** — the consequence of being unable to ask for
2.6 objects. At target 2 the bias is 94 % of the error: the prompt "one" is the closest integer
and it is wrong by almost exactly one object every time. A continuous perturbation of the noise is
precisely the instrument that removes a bias of 0.93 without touching the spread, and that is the
clearest statement of what the sampler is for that this project has produced.

### 2026-11-25 — the spread is not noise; it is a deterministic function of the sampler's own inputs

Variance of the counted value at fixed n, `eval_baseline_white_s8`, decomposed:

| | share of variance |
|---|---|
| seed (which ε was drawn) | 14.7 % |
| (colour, shape) | 21.0 % |
| remainder | 64.4 % |

The remainder is not replication noise, because **there is no replication**: every
(seed, colour, shape, n) cell holds exactly one image, and the frozen generator is deterministic
given z and the prompt. The counted value is a function `f(ε, colour, shape, n)` evaluated once.
The 64.4 % is the seed × cell interaction, and Δφ(ε, n, c̄) observes **all three** of those inputs.

So "spread" is the wrong mental model for what the sampler is up against. It is not irreducible
noise setting a floor on achievable MAE; it is a deterministic function of exactly the arguments
Δφ is given, and an expressive enough Δφ with enough budget could in principle invert it to zero
error. The real limits are the two this session measured: how far z may move before the image
collapses (0.08 along a learned direction, `direction_probe.py`), and how much count change is
available inside that radius (`leverage_probe.py`, running).

This also corrects how the earlier ceiling analysis should be read. "Skill turns positive when the
spread halves" is not a claim about beating noise — it is a claim about how much of `f` the
sampler has to invert, and the obstacle is the perturbation budget, not stochasticity.

**Spec §8's per-seed question, answered.** Spearman(requested n, counted) per seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| ρ | +0.866 | +0.830 | +0.643 | +0.718 | +0.778 | +0.737 | +0.644 | +0.771 | **+0.748** |
| mean counted | 9.68 | 10.03 | 8.13 | 8.35 | 7.76 | 10.66 | 8.01 | 11.66 | |

Every seed follows the count, and none has a fixed preferred value — the §8 worry that seeds carry
their own count, which would have to be overcome before anything else, does not apply here. The
seeds differ in *level* (7.76 to 11.66) far more than in whether they respond, which is the same
gain-and-offset picture the pooled fit gave.

### 2026-11-25 — the leverage probe's small-radius numbers were an artifact; caught before use

The first run of `--mode leverage` printed:

```
  d(z,start)   mean |count - start|   img sd
  <=    0.05                   1.89     51.1  scene
  <=    0.10                   1.89     51.1  scene
  <=    0.15                   1.89     51.1  scene
  <=    0.20                   4.00     52.7  scene
```

Read naively that says a displacement of only 0.05 — well inside the 0.08 safe radius — already
moves the count by 1.89 objects, which would have settled the feasibility question in favour of
the premise. **It is an artifact, and the three identical values are the tell.**

The slerp path has 9 points at t = 0, 0.125, …, 1. Two independent Gaussians are nearly orthogonal
in 16384 dimensions, so the first step already lands at **d = 0.196**:

```
t      : 0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000
d(z,a) : 0.000 0.196 0.390 0.581 0.766 0.944 1.113 1.271 1.418
```

Nothing was sampled below 0.196 at all. The bins at 0.05, 0.10 and 0.15 contain exactly one point
— **index 0, the start itself** — which survives the `d > 0` filter only because slerp at t = 0
returns the start up to float error rather than exactly. So those rows compare the start point
with itself, and should read 0.

They read 1.89 because of a second problem: the start's count is measured twice, once in the
12-candidate scan (`c_lo`) and once as `path[0]`, and the two disagree. Blue pentagon n = 8 gives
`c_lo` = 1 and `path[0]` = 17; red cross n = 8 gives 1 and 2. Over the nine paths that is
(0+0+16+1+0+0+0+0+0)/9 = **1.89**, reproducing the number exactly.

**Two things to fix, and one new question.**

1. The path needs points at the radii that matter. The safe learned radius is 0.08, below the
   probe's *smallest* measurement, so the probe as run says nothing about the regime in question.
   Spacing t as {0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15, 0.25, 0.5, 1} puts d at roughly
   {0, 0.016, 0.031, 0.055, 0.078, 0.11, 0.157, 0.235, 0.39, 0.77, 1.42}.
2. Count change must be measured against `path[0]` from the *same* pass, not against `c_lo` from a
   different one.
3. **Why do the two passes disagree at all?** z is identical and the frozen generator runs under
   `no_grad`. The one difference is batch composition — the scan decodes 12 images in chunks of
   8 + 4, the path decodes 9 in chunks of 8 + 1 — so the same latent sits at a different position
   in a different-sized batch. If that can move a count from 1 to 17, it is a measurement hazard
   for every number in this log, and it is worth a direct test rather than an assumption.

What the probe does still support, at the radii it actually sampled: at d = 0.196 the count has
moved about 4 objects with image sd 52.7 (healthy), and the endpoints differ by 10.9 objects at
d = 1.42. Count-changing directions that keep the image valid certainly exist. Whether any of that
is reachable inside 0.08 remains **unmeasured**.

### 2026-11-25 — counting resolution is a first-order effect, and the NMS fix was over-credited

`resolution_probe.py`, 960 paired images from `eval_baseline_fix` (bare-noun template), counting
the *same* saved grids three ways:

| counting | mean | MAE | slope n ≥ 4 | spread | zero % |
|---|---|---|---|---|---|
| 512 px, IoU-only — what `records.csv` holds | 10.88 | 6.956 | +0.624 | 7.33 | 1.8 |
| 256 px, IoU-only — **resolution changed, rule unchanged** | 8.55 | 5.229 | +0.362 | 5.66 | 2.6 |
| 256 px, containment — **both changes** | 5.78 | 4.287 | +0.293 | 5.04 | 2.6 |

| isolated effect | mean | slope |
|---|---|---|
| resolution alone (512 → 256, same NMS) | **−2.33** | **−0.262** |
| NMS rule alone (both at 256 px) | −2.77 | −0.069 |

**Only 29.9 % of images receive the same count at 512 px and 256 px under the identical rule.**
Resolution is not a detail; it is comparable in size to the suppression rule and larger than it on
the slope, where it accounts for about four fifths of the change.

**The correction.** The containment-NMS entry reported the baseline slope moving +1.031 → +0.595
and credited the suppression rule. That comparison was 256 px grid counts against a 512 px
`records.csv`, so it contained both changes, and on this decomposition the resolution is the bigger
part of the slope move. The nested-box argument itself is unaffected — it rests on the geometry of
IoU for a small box inside a large one, and on the visibly 2×2 cross image — and the rule still
removes 2.77 objects per image on its own. But the *magnitude* previously attributed to it was
roughly double what it earns.

**What this does and does not touch.**

- Every comparison in this session is internally consistent: `eval_baseline_white_s8` and each
  candidate run are counted by `count_grids.py` on 256 px grids with containment NMS. The skill
  numbers (raw −0.853, remapping +0.016, affine ceiling +0.341) are like-for-like and stand.
- The **value** 1.514 in `counted = 1.514 n + 0.069` is a property of (generator, counter,
  resolution), not of the generator alone. At 512 px the detector finds more objects and the
  bare-noun slope is about 1.7× steeper, so the same fit at 512 px would report a larger gain.
  The *structure* — a near-zero offset and a constant multiplier — is what the argument rests on,
  and that is what a sampler has to correct, whichever counter defines the target.
- Which resolution is "right" is not settled here, and nothing in this log should be read as
  settling it. OWLv2 rescales its input to 960 px, so a 256 px cell is interpolated 3.75× and a
  512 px cell 1.875×; the sharper input plausibly detects better, which would make 256 px an
  under-count. Deciding it needs ground-truth counts, which these images do not have.

`eval.grid_scale: 1.0` now makes the two agree by construction for new runs, and
`scripts/count_grids.py` carries the warning. Runs counted at different resolutions must not be
compared, which is exactly the mistake this entry corrects.

### 2026-11-25 — looking at the images: the over-drawing is real, and the counter fails on dense scenes

`runs/figures/countable_gain_images.png`, countable-scene prompts, seed 1. Two things are visible
that no table showed.

**The over-drawing is the generator's, not the detector's.** Red cross at n = 3 genuinely contains
about eight crosses; at n = 4, about nine. The counter is not inventing objects. And n = 1 is
exact for all three shapes — one object, counted 1 — which is the cleanest confirmation that the
count reaches the model at all.

**The counter collapses on crowded images.** Blue pentagon at n = 10 is a dense field of roughly
twenty shapes and OWLv2 counted **1**. This is not rare:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 1.01 | 3.57 | 5.37 | 6.99 | 8.79 | 10.62 | 11.83 | 12.84 | 15.82 | 16.03 |
| sd | 0.34 | 3.26 | 2.72 | 3.49 | 3.81 | 3.59 | 4.22 | 5.46 | 6.26 | 8.00 |
| **median** | **1** | **3** | **5** | **6** | **8** | **10** | 11 | 12 | 15 | 15 |
| counted < n/2 | 2.9 % | 1.5 % | 5.9 % | 8.8 % | 8.1 % | 3.7 % | 5.9 % | 8.8 % | 4.4 % | 13.2 % |

**8.3 % of n ≥ 8 rows return 2 objects or fewer.** Dropping the 30 rows with n ≥ 6 and counted ≤ 2
(of 680) changes the fit from `1.514 n + 0.069` to `1.754 n − 0.658` and raises held-out skill
from +0.341 to **+0.421**. A handful of catastrophic under-counts was holding the estimated gain
down, which means **1.5× was itself an underestimate** of how much the generator over-draws.

**The median is the cleaner description.** It is exactly `2n − 1` for n ≤ 3 and `2n − 2` for
n = 4–6, then falls progressively behind as both the generator and the counter saturate. Inverting
the median curve, fitted on even seeds and scored on odd, gives **MAE 1.479, skill +0.408** — a
little better than the linear affine ceiling and far more interpretable. The frozen generator's
step is about **two objects per unit of requested n**.

**Which sharpens what the sampler is for.** Choosing, for each target, the prompt whose median
output is closest:

| target | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| best prompt | 1 | 1 | 2 | 2 | 3 | 4 | 4 | 5 | 5 | 6 |
| its median | 1 | 1 | 3 | 3 | 5 | 6 | 6 | 8 | 8 | 10 |
| | exact | −1 | exact | −1 | exact | exact | −1 | exact | −1 | exact |

Six of ten targets are reachable exactly; the other four sit one object away, because the
generator's own step is two and the integer prompt grid can only land on every other count.

The earlier claim that the residual is "mostly bias, not spread" needs splitting by range. At low
targets it is bias — the generator draws one object for "one" with sd 0.34, so using that prompt
for a target of 2 is wrong by one almost every time. At high targets it is spread — prompt 6 has
the right median for target 10 but a sd of 3.7. A continuous perturbation addresses the first
directly; the second is the harder half.

### 2026-11-25 — what ε-conditioning is worth: an oracle upper bound

Δφ(ε, n, c̄) receives the noise, the count and the pooled text embedding, and the frozen generator
is deterministic, so `counted = f(ε, colour, shape, n)` is a function of exactly those arguments.
How much is it worth to know ε? On the 136 (colour, shape, seed) cells of
`eval_baseline_white_s8`, each of which has a counted value for every n = 1…10:

| | MAE | skill |
|---|---|---|
| fixed global prompt map, **in sample** | 2.077 | +0.169 |
| **oracle: per-ε choice among the 10 available outputs** | **0.951** | **+0.619** |

*(The same fixed map scored on held-out seeds gives +0.016; both numbers above are in-sample, so
the gap between them is the like-for-like comparison.)*

Knowing ε is worth about **+0.45 skill** — far more than the choice of prompt map. Some prompt
lands exactly on the target for **41.2 %** of (cell, target) pairs:

| target | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| exact hit available | **95.6 %** | 33.1 % | 25.7 % | 39.0 % | 34.6 % | 41.2 % | 37.5 % | 30.9 % | 33.1 % | 41.9 % |

Target 1 is nearly always reachable, which is the same fact as `1.01 ± 0.34` seen from another
angle. For every other target, roughly a third of noises already have *some* prompt that produces
exactly the right count — the information is in the noise, and a fixed map cannot use it.

**Two reasons this is an upper bound and not a target.** The oracle reads the counted value it is
trying to predict, so it is optimistic by construction. And the sampler cannot choose among ten
prompts: it must *move the noise*, within a radius where the image survives, which
`direction_probe.py` put at 0.08 for a learned direction. Whether the needed motion fits inside
that radius is the question `--mode leverage` is running now.

What the bound does establish is that the architectural choice — conditioning Δφ on ε rather than
on n alone — is where the available headroom is. A count-independent perturbation, which is what
every Method D run actually learned (`d_pair` 0.00027 against an anchor displacement of 0.128),
forfeits all of it.

### 2026-11-25 — the detector threshold, settled by looking

Re-counting the same 1360 images with OWLv2's score threshold raised from 0.10 to 0.25:

| threshold | fitted gain | raw skill | remapped, per shape | affine ceiling | median ceiling | zero % |
|---|---|---|---|---|---|---|
| 0.10 | `1.514 n + 0.069` | −0.853 | **+0.082** | +0.341 | +0.408 | 0.7 |
| 0.25 | `1.033 n + 0.546` | **−0.412** | **+0.062** | +0.062 | +0.236 | 9.9 |

Taken at face value this says the higher threshold is better: the baseline's skill nearly doubles
and the fitted gain becomes 1.03, i.e. the generator would be drawing almost exactly the requested
number. **It is the degeneracy trap in a new place.** A higher threshold finds fewer objects, and
since the generator over-draws, missing some brings the count closer to n — two errors cancelling,
not better counting. The zero rate rising 0.7 % → 9.9 % is the tell, and a threshold tuned on
baseline skill would be tuned to under-detect.

**Settled by counting the images by hand.** Eight cells, chosen for the largest disagreement
between the two thresholds (`runs/figures/ground_truth_check.png`):

| asked n | counted by eye | OWLv2 @0.10 | OWLv2 @0.25 |
|---|---|---|---|
| 1 | **1** | 1 ✓ | **0** |
| 2 | ~14 | 17 | 8 |
| 3 | ~12 | 11 ✓ | **1** |
| 4 | ~4–5 | 6 | **0** |
| 5 | ~8–9 | 9 ✓ | 2 |
| 6 | ~11–13 | 14 | **1** |
| 8 | ~11–13 | 13 ✓ | 4 |
| 10 | ~16–18 | 17 ✓ | **0** |

At 0.25 the detector returns 0, 1 or 2 on images that visibly hold ten or more objects. **0.10 is
the right operating point** and `counted = 1.514 n + 0.069` stands as the better estimate of the
generator's gain. Two honest qualifications: at 0.10 the counts still run slightly above a manual
count (17 against ~14, 6 against ~4–5), so a little over-counting probably remains; and these
eight were *selected* for maximum threshold disagreement, so they skew dense and are not a random
sample of the evaluation.

**What survives either way.** Per-shape prompt remapping scores **+0.082 at threshold 0.10 and
+0.062 at 0.25**, and global remapping +0.016 and +0.080. The generation-side result does not
depend on the operating point; it is the *ceilings* that do (affine +0.341 → +0.062), because at
0.25 the apparent gain is already 1 and a calibration has nothing left to correct — what remains
there is spread, which no affine map can touch.

### 2026-11-25 — the density counter never trained: softplus at bias −10 is a dead zone

Method B's counter completed 30 epochs and learned **nothing**. The validation history is flat
from the first epoch to the last:

| epoch | `mae_all` | `exact_all` | `mae_present` | `exact_present` | `stray_mass` |
|---|---|---|---|---|---|
| 0 | 0.2084 | 0.9618 | 5.3115 | **0.0000** | 0.0059 |
| 9 | 0.2030 | 0.9618 | 5.3174 | **0.0000** | 0.0000 |
| 29 | 0.2029 | 0.9618 | 5.3174 | **0.0000** | 0.0000 |

Across all 30 validations `val/mae_present` takes 7 distinct values spanning 5.3115 to 5.3174.

**The counter predicts zero everywhere.** `exact_all` of 0.96 looks healthy and is the trap: 15 to
17 of the 18 categories are absent in any image, so predicting nothing is correct for almost every
channel. `mae_present` of 5.317 is simply the mean count of the categories that *are* present —
the error you get by outputting nothing — and `exact_present` of exactly 0.0000 at every epoch
says it never once got a present category right.

**The cause is the head initialisation.** The output is `softplus`, whose gradient is
`sigmoid(bias)`:

| bias | softplus | × 4096 px | gradient | vs −10 |
|---|---|---|---|---|
| **−10** | 0.000045 | **0.19** | **0.000045** | 1× |
| −8 | 0.000335 | 1.37 | 0.000335 | 7× |
| −7 | 0.000911 | 3.73 | 0.000911 | 20× |
| **−6** | 0.002476 | **10.14** | 0.002473 | **54×** |
| −4 | 0.018150 | 74.34 | 0.017986 | 396× |

At −10 the head begins predicting 0.19 objects per channel with its gradient attenuated 22 000×.
It cannot escape, and because the loss is dominated by empty channels there is nothing pulling it
out: all-zeros is a strong local optimum reached immediately. The comment in `configs/counter.yaml`
justified the value correctly — `softplus(0)` summed over the 64×64 map is 2839 spurious objects
per channel — but overshot into a region where the layer does not learn.

**Fixed**: `head_bias_init: -10.0 → -6.0`, which starts at 10.1 objects per channel against a true
mean of about 5 for present categories, with 54× the gradient; and `l1_weight: 0.1 → 1.0`, because
the density MSE is dominated by the empty channels and the term that actually measures *count*
error needs weight. The latent cache is keyed on the dataset config only — model, resolution,
categories, seed, radius range, dataset and augmentation — so neither change invalidates it and
the retrain costs the training loop alone, about 18 minutes, rather than another 35 of rendering
and VAE encoding.

**This is the third diagnostic in this project that looked healthy during total failure**, after
`geom/d_anchor_*` reading 0.13–0.22 inside a δ = 0.3 budget while the decode was a flat colour
block, and Method C's MAE improving by 2.855 while it emitted 0.19 objects per image. The common
shape: an aggregate that averages over a majority class — absent categories here, empty images
there — and so reports the majority's success while the minority that matters fails completely.
`val/exact_present` was the number that told the truth, and it was in the log from epoch 0.

### 2026-11-25 — batch composition does perturb generation, but only by ±1 object

The same latent decoded in batches of 1, 2, 4, 8, 9 and 12, with the frozen generator under
`no_grad`. Every column should be identical and they are not:

| prompt | 1 | 2 | 4 | 8 | 9 | 12 | latent sum | spread |
|---|---|---|---|---|---|---|---|---|
| blue pentagon n=2 | 3 | 3 | 3 | 3 | 3 | 3 | 436.3 | 55.3 |
| red cross n=5 | 7 | 7 | 7 | 7 | 7 | 7 | −312.2 | 111 |
| **red cross n=8** | 10 | 10 | **9** | 10 | 10 | 10 | −1014.9 | 51 |
| **yellow diamond n=2** | **2** | **3** | **3** | **2** | **2** | **2** | 3680.6 | 126 |
| yellow diamond n=8 | 17 | 17 | 17 | 17 | 17 | 17 | 3119.5 | 33.8 |

The latent sums differ in every row, so generation is genuinely not bit-identical across batch
sizes — `generate_latents` chunks at 8, so a latent sits in a chunk of 1, 4, 8 or 12 depending
only on how many images were asked for, and bf16 reductions over different batch shapes do not
agree to the last bit. **But the counts move by exactly ±1, in 2 of 9 prompts.**

**So batch jitter does not explain the leverage probe's 1 against 17.** A ±1 effect cannot produce
a 16-object discrepancy. The explanation is in that probe's own output: blue pentagon n = 8 walked
`[17, 1, 18, 18, 18, 1, 1, 22, 26]` — ones scattered between seventeens and twenty-sixes. Those
are not intermediate counts, they are the **detector's dense-scene failure**, the same mode that
returns 2 objects or fewer on 8.3 % of n ≥ 8 rows. On a crowded image OWLv2 is **bistable**: it
either resolves the field into many boxes or collapses onto one, and a perturbation far too small
to change what is drawn is enough to flip it between the two.

Two consequences worth carrying forward:

- **±1 is the error bar on any single count** from batch-composition jitter alone. It is small
  against a spread of 3–8 and does not threaten any conclusion in this log.
- **The dense-scene failures are noise, not a fixed property of an image.** The same latent can
  read 17 or 1. That makes them averageable — more seeds genuinely helps — but it also means a
  per-image count at high n should never be trusted on its own, and the 8.3 % figure is a rate,
  not a set of identifiable bad images.

The second point is the one that matters for method comparison: a run evaluated at high n is being
scored partly on a coin flip, which is another reason `results_table.py` leads with spread and the
zero rate rather than with MAE.

### 2026-11-25 — the premise is feasible: the leverage needed is available inside the safe radius

`--mode leverage`, refitted with the sample points bunched near zero and the count compared
against each path's own start. Nine slerp paths between pairs of noises the frozen model turns
into very different counts for the same prompt; spherical interpolation holds ‖z‖ constant, so
every point satisfies `L_pri` and is as Gaussian as its endpoints — the path cannot collapse, and
the flat image sd column confirms it.

| d(z, start) | median \|Δcount\| | ≥ 1 obj | ≥ 2 obj | img sd |
|---|---|---|---|---|
| 0.016 | 0 | 11 % | 11 % | 51.0 |
| 0.032 | 0 | 44 % | 11 % | 51.1 |
| 0.055 | 1 | 67 % | 33 % | 51.2 |
| **0.079** | **1** | **67 %** | **44 %** | **51.4** |
| 0.110 | 2 | 89 % | 56 % | 51.6 |
| 0.236 | 3 | 89 % | 56 % | 53.4 |
| 0.768 | 5 | 78 % | 78 % | 59.4 |
| 1.418 | 10 | 100 % | 100 % | 60.2 |

**At d = 0.079, just inside the 0.08 radius where a learned direction still draws a scene, the
median count change is 1 object, two thirds of paths have moved by at least one and 44 % by at
least two — with image sd 51.4 against 51.0 at the start.**

**One object is exactly the correction that is needed.** The targets prompt remapping cannot reach
— 2, 4, 7 and 9 — are each off by precisely one object. The leverage available inside the safe
radius and the leverage the task requires are the same size. The premise is not chasing something
that does not exist.

Three honest qualifications. These are **slerp paths, not learned directions**: the walk is toward
another noise that happens to give a different count, which is a lower bound on what an optimised
direction could achieve per unit distance, but also a direction that stays on the Gaussian
manifold, which the learned ones demonstrably do not. The response is **extremely path-dependent**
— blue pentagon n = 2 goes 1 → 22 by d = 0.055 while blue pentagon n = 5 sits at 8 → 7 until
d = 0.768 — so a median of 1 hides paths that move not at all and paths that move twenty. And the
mean column is unusable: blue pentagon n = 8 walks `17, 1, 18, 18, 1, 1, 1, 1, 18, 17, 26`, the
detector's dense-scene bistability, and a single 16-object flip adds 16/9 = 1.78 to the mean by
itself — which is exactly the d = 0.016 row, and exactly the artifact that made the first run of
this probe report a false 1.89.

So the obstacle for Methods A, C and D was never that the budget is too small for the task. It is
that the direction those losses choose destroys the image before it uses the budget — measured at
4× smaller radius than a random direction — and, for Method D, that the perturbation it learned
was the same for every count (`d_pair` 0.00027 against an anchor displacement of 0.128).

### 2026-11-25 — template search: "a flat lay photo of …" wins, "exactly" and digits hurt

Six phrasings on the frozen model, 9 (colour, shape) cells × 3 seeds × 10 counts each, the prompt
for every target chosen on seeds 0 and 2 and scored on seed 1:

| template | slope | spread | gain a | offset b | zero % | raw skill | remapped |
|---|---|---|---|---|---|---|---|
| `{count} {color} {shape} on a white background` (current) | +1.802 | 5.23 | 1.980 | +1.180 | 0.0 | −1.862 | −0.004 |
| `{count} small {color} {shape} …` | +2.472 | 5.84 | 2.494 | −0.765 | 1.1 | −2.206 | +0.022 |
| `exactly {count} {color} {shape} …` | +1.642 | 6.64 | 2.025 | +1.635 | 0.7 | −2.292 | **−0.209** |
| `{count}` as **digits** (`3` not `three`) | +2.438 | 6.52 | 2.363 | +0.523 | 0.7 | −2.412 | **−0.209** |
| `{count} {color} {shape} spread out on a plain white background` | +2.111 | 4.79 | 2.193 | −0.965 | 3.7 | −1.615 | +0.164 |
| **`a flat lay photo of {count} {color} {shape} on a white background`** | +2.015 | **3.79** | 2.085 | −1.207 | 1.1 | **−1.179** | **+0.231** |

**Flat-lay wins on every column that matters**: the best remapped skill (+0.231 against the
current template's −0.004), the lowest spread (3.79 against 5.23) and the best raw skill. "Spread
out on a plain white background" is second at +0.164. Both winners push the model toward a
*photographic layout of separate objects* rather than a texture, which is the same lever the
countable-scene template pulled in the first place.

**Two that were expected to help and did the opposite.** `exactly {count}` is the worst of the six
and raises spread to 6.64 — the word that most directly signals precision makes the count *less*
reliable, so SD-Turbo's text encoder takes nothing useful from it. **Digits instead of words**
(`data.count_format: digit`) are equally bad at −0.209; that config option has sat untested since
the project began and it hurts.

**The comparison is within-probe only.** 9 cells and 3 seeds, held out on one seed, is far thinner
than the full 17-cell 8-seed evaluation, and the absolute numbers run well below it — the current
template reads −0.004 here but scores +0.016 global and **+0.082 per shape** on the full run. So
+0.231 **cannot** be set beside +0.082; only the ranking transfers. The full flat-lay evaluation,
on the same cells and seeds as `eval_baseline_white_s8` and scored through `remap_prompts.py` and
the paired bootstrap, is what produces a comparable number, and it is generating now.

This is the third time prompt phrasing has moved the result more than any trained method has.

### 2026-11-25 — the counter's collapse is class imbalance, not initialisation

Retraining with `head_bias_init: -10 → -6` and `l1_weight: 0.1 → 1.0` **did not fix it**. By epoch
8 `val/exact_present` was still exactly 0.0000 and `val/mae_present` still 5.3168 — the same dead
run, with the configuration confirmed applied in the saved `runs/counter/config.yaml`.

The arithmetic says why. The counter has **48 channels** — 6 colours × 8 shapes, not the 18 the
previous entry assumed — and `max_categories_per_image` is 3, so **46 of 48 channels are zero in
any image**. Predicting zero everywhere scores a training L1 of 2 × 5 / 48 = **0.208**, and the
observed L1 was **0.201**. That is not "near the all-zeros solution"; it *is* the all-zeros
solution, to three figures.

**Raising the head bias made the pull stronger, not weaker.** At bias −6 every channel starts at
10.1 objects, so the initial L1 is about (2 × 5.1 + 46 × 10.1) / 48 ≈ 9.9 — an overwhelming
gradient pointing at zero for 96 % of the output. The optimiser takes it, drives the bias back
below −10, and re-enters softplus's dead zone. The previous entry diagnosed the dead zone
correctly and then prescribed the wrong remedy: the dead zone is where the model *ends up*, not
why it goes there.

**Fixed where the cause is.** `counter_losses()` now up-weights channels whose true count is above
zero, in both the density MSE and the count L1, by `counter.train.present_weight` (20.0). The
present channels carry **46.5 %** of the loss instead of 4.2 %. The loss moved out of the training
loop into a function so it can be tested, and four tests pin it: that predicting zero now costs
200/86 instead of 10/48, that a perfect prediction scores zero, that `present_weight = 1`
reproduces the original unweighted loss exactly, and that a stray prediction in an empty channel
is still penalised. 142 tests pass.

**The honest tally on this bug.** Two full training runs, roughly 50 minutes of GPU, spent
producing a counter that predicts nothing — and the first one ran to completion with
`val/exact_all = 0.9618` on screen the whole time. That is the fourth diagnostic in this project
to report success during total failure, and the second time the *same* number misled me, because
after the first collapse I fixed the symptom I could see rather than checking whether the
arithmetic of the all-zeros solution matched the loss I was reading. It did, exactly, and one
division would have shown it.
