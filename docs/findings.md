# Findings

What is established, with the measurement behind each claim. The running narrative, including the
wrong turns and the corrections, is in [experiment_log.md](experiment_log.md); this file is the
summary a reader should start from.

Every number here is scored with the **skill score** against the best single constant predictor:

```
skill = (MAE_best_constant - MAE) / MAE_best_constant
```

For n drawn uniformly from 1..10 the best constant predicts 5 or 6 and scores MAE 2.500, so
**skill 0.000 means "no better than ignoring the prompt"** and a negative skill means worse than
that. A raw MAE cannot establish that counting works, which is why it never leads a table here.

All counts come from OWLv2 (threshold 0.1, no colour in the query, containment NMS) on the 256 px
saved grids. Results fitted on even seeds are always scored on odd seeds.

---

## 1. What beats the baseline

| | MAE | skill | kind |
|---|---|---|---|
| frozen baseline, bare-noun prompts | 5.210 | **−1.084** | |
| frozen baseline, countable-scene prompts | 4.632 | **−0.853** | prompt |
| global prompt remapping | 2.460 | **+0.016** | prompt, held out |
| **per-shape prompt remapping** | **2.296** | **+0.082** | prompt, held out |
| per-(colour, shape) remapping | 2.485 | +0.006 | **overfits** |
| — measurement-side affine correction | 1.649 | +0.341 | not generative |
| — measurement-side median inversion | 1.479 | +0.408 | not generative |

**Per-shape prompt remapping at +0.082 is the best generation-side result**, and the first thing in
this project to beat the constant predictor by changing what is generated rather than how it is
scored. It needs no training. Per-(colour, shape) has 170 free choices against 680 training rows
and does worse, which is how the granularity was chosen rather than assumed.

The last two rows are **not methods**. They correct the counted value after the fact and are
recorded as the ceiling a generative method could aim at.

**No trained sampler has beaten the baseline.** Method C emitted **0.19 objects per image** with
82 % of images empty — its MAE of 5.311 is essentially the score for predicting zero every time,
which is why `results_table.py` marks such runs degenerate rather than crediting the MAE. (That
5.311 was counted at 512 px against a 256 px baseline, so the two numbers are not strictly
comparable — see §5 — but a run producing 0.19 objects per image is degenerate at any resolution,
and that is what the conclusion rests on.) Both Method D variants destroyed the generator (§4).

---

## 2. The frozen generator already hears the count

With countable-scene prompts (`"{count} {color} {shape} on a white background"`), fitted on
held-out seeds:

```
counted = 1.514 n + 0.069          (1.754 n - 0.658 once the counter's dense-scene failures are excluded)
```

The offset is zero. The model is not failing to hear the number — it hears it and draws about
twice too many. The median is the cleaner statement: exactly `2n − 1` for n ≤ 3 and `2n − 2` for
n = 4–6, then falling behind as generator and counter saturate.

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| median counted | 1 | 3 | 5 | 6 | 8 | 10 | 11 | 12 | 15 | 15 |
| mean | 1.01 | 3.57 | 5.37 | 6.99 | 8.79 | 10.62 | 11.83 | 12.84 | 15.82 | 16.03 |
| sd | 0.34 | 3.26 | 2.72 | 3.49 | 3.81 | 3.59 | 4.22 | 5.46 | 6.26 | 8.00 |

**Asked for one object it draws 1.01 ± 0.34.**

The prompt template matters more than any method tried. It also repairs an inverted response:
pentagon's slope over n ≥ 4 is **−0.645** with bare-noun prompts and **+1.029** with countable
ones. With the bare template the model answered "ten pentagons" with edge-to-edge tiling, for
which the true count is not well defined.

---

## 3. What the sampler is actually for

The generator's step is about **two objects per unit of n**, so an integer prompt grid can only
land on every other count. Choosing the best prompt per target:

| target | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| its median output | 1 | **1** | 3 | **3** | 5 | 6 | **6** | 8 | **8** | 10 |

Six of ten targets are reachable exactly; four sit one object away and **no prompt can ask for 2.6
objects**. A perturbation of the initial noise is continuous, and that gap is what it is for.

The error splits by range. At low targets it is **bias**: the generator draws one object for "one"
with sd 0.34, so using that prompt for a target of 2 is wrong by one almost every time — removing
a systematic 0.93 without touching the spread is exactly what a continuous knob does. At high
targets it is **spread**: prompt 6 has the right median for target 10 but a sd of 3.7. The second
is the harder half.

**The spread is not noise.** The frozen generator is deterministic given z and the prompt, and
every (seed, colour, shape, n) cell holds one image, so there is no replication anywhere. The
variation at fixed n is a deterministic function `f(ε, colour, shape, n)` — and Δφ(ε, n, c̄)
observes all of those arguments. Nothing about the spread is irreducible; the limits are the
budget and the optimiser, not stochasticity.

---

## 4. Why the attention-based methods failed

**`L_anc` constrains the wrong quantity.** `geometry.delta` bounds ‖z_n − ε‖/√D, and that was
measured not to control whether the output is a scene:

| d(z, ε) | random direction | learned direction |
|---|---|---|
| 0.00 | 4.50 objects, 0 % empty, img sd 51.9 | 4.50, 0 %, 51.9 |
| 0.08 | 4.36, 0 %, 51.9 | 2.86, 16.7 %, 33.1 |
| 0.13 | 4.56, 0 %, 51.9 | 1.25, **52.8 %**, 17.4 |
| 0.30 | 4.92, 2.8 %, 54.5 | 0.14, **86.1 %**, 10.1 |

A **random** displacement at the full δ = 0.3 budget leaves the image essentially unchanged. A
**learned** one stops drawing scenes past **0.08** — a 4× asymmetry, the ordinary adversarial one:
the generator denoises isotropic perturbation back toward the data manifold, and gradient descent
finds the thin subspace that does not get denoised away. **No choice of δ fixes this**, because
any budget large enough to let the sampler act is large enough to let it destroy the image.

This explains both Method D runs exactly. `d_white_clip200` held `d_anchor` at 0.13–0.22 — between
52.8 % and 86.1 % empty on that table — and drew solid black squares while `L_anc` read 0.0 all
run. `d_white_clip1` held 0.045–0.089, straddling the edge, which is why its images were
recognisable but poor. **`grad_clip: 1.0` was not a tuning accident; it was the only thing keeping
that run inside the safe radius**, and raising it to 200 removed the last constraint doing any work.

Two consequences, both now in the code:

- `geometry.log_x0: true` logs `geom/d_x0_mean|max`, the distance moved in **x̂0** space. It is the
  only diagnostic that shows a collapse — the existing geometry diagnostics reported a healthy run
  throughout total failure.
- `L_x0anc` (`geometry.delta_x`) hinges on that distance instead of on ‖z − ε‖.

**`L_ord` had never been active.** `configs/method_d.yaml` stages the ordering term at step 3000
and every Method D run was 3000 steps, so the one term requiring the perturbation to *differ* per
count never entered the loss. `loss/ord` was logged every step at 0.41–0.45, about 1000× its own
target, because the method computes every term and sums only the active ones. Turning it on worked
on its own terms — `d_pair` rose 0.0025 → 0.105 in 900 steps against 0.00027 at the end of the
previous run — but it was applied on top of an already collapsing generator. `configs/method_b.yaml`
carries the identical staging.

---

## 5. Measurement caveats that change numbers

**Counting resolution is first-order.** `scripts/evaluate.py` counts the full 512 px decode but
saves grids at `eval.grid_scale` (0.5 by default). On 960 paired images, **only 29.9 % receive the
same count at 512 px and 256 px under the identical rule.** Resolution alone moves the mean by
−2.33 and the slope by −0.262; the containment NMS rule moves them −2.77 and −0.069. An earlier
entry credited a slope change of +1.031 → +0.595 entirely to the NMS rule; about four fifths of it
was resolution. Runs counted at different resolutions must not be compared.

**The counter fails on dense scenes.** 8.3 % of n ≥ 8 rows return two objects or fewer, on images
that visibly contain fifteen or more. Excluding them raises the estimated gain from 1.514 to 1.754
and held-out skill from +0.341 to +0.421 — **so the gain is if anything understated.**

**Four detector defects were found and fixed**, all of which inflated the baseline: proper-noun
queries ("blue cross" returned nothing on 90 % of images), a threshold at 3× the library default,
a padded-coordinate normalisation bug, and nested-box double counting that plain IoU cannot
suppress, since a small box inside a large one has IoU = a/A.

**Degenerate runs flag themselves.** `scripts/results_table.py` leads with spread, per-shape slope,
counted mean and zero rate, and marks a run degenerate rather than letting a good MAE speak for it.
This exists because Method C improved MAE by 2.855 against the baseline it was measured on,
significant on a paired test, while producing 0.19 objects per image.

---

## 6. Open

- **Is usable count change reachable inside the safe radius?** At d = 0.196 the count moves about
  4 objects with the image intact, and the endpoints of a slerp path differ by 10.9 objects at
  d = 1.42 — so count-changing directions that keep the image valid certainly exist. Whether any
  is reachable inside 0.08 is unmeasured; the first probe sampled nothing below 0.196.
- **Does batch composition change a count?** The same latent was counted as 1 in one pass and 17 in
  another, differing only in how many images shared the batch. If real, every number here carries
  an error bar.
- **Method B** — the only method whose loss reads the decoded image — has never run, because its
  counter has never been trained. A flat colour block contains zero objects for a density counter,
  so it should be structurally resistant to the collapse that destroyed Method D.
