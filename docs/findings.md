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

## 1. The research direction, and what the prompt numbers are for

**The count must leave the prompt.** With the number in the text, CLIP already does part of the
counting, so no experiment can separate the sampler's contribution from the text encoder's — the
result is uninterpretable whichever way it comes out. The direction of this work is therefore the
`data.include_count: false` setting: the prompt reads `"a flat lay photo of blue crosses on a
white background"` for **every** n, identical string, unconditional plural so n = 1 does not leak.
The frozen generator then produces the same image for every count and its response is **flat by
construction, slope exactly 0**, so any skill above the constant predictor belongs to Δφ alone.
Pass or fail, with nothing to hide behind.

### Prompt remapping is a measurement, not a method

Earlier work here found that issuing a *different* number than the one wanted scores well:
per-shape remapping reaches skill **+0.082** on the countable template and **+0.299** on the
flat-lay one, against a raw baseline of −0.853, verified by paired bootstrap. **That is recorded
as a diagnostic of how broken the count channel is, and is not a direction this project should
follow.** The honest reading is damning rather than encouraging: *lying to the model about the
number works better than asking it correctly.*

It is unacceptable as a method for concrete reasons, not only aesthetic ones:

- it repairs nothing — it exploits a measured bias while the architecture stays exactly as wrong;
- it is a per-shape lookup table that **cannot extrapolate** to a new shape without refitting;
- it is **quantised** — 4 of 10 targets are unreachable because the generator's step is about two
  objects and an integer prompt grid can only land on every other count;
- and it keeps the number in the prompt, which is the contamination above.

What it does establish, and what is worth carrying forward, is the *size and shape of the defect*:
`counted ≈ 1.5–2 n` with a **near-zero offset**, so the count reaches the model and is then scaled.
That is a calibration failure of a channel that works, which is a far more tractable problem than
a channel that does not.

| | MAE | skill | kind |
|---|---|---|---|
| frozen baseline, bare-noun prompts | 5.210 | −1.084 | |
| frozen baseline, countable-scene prompts | 4.632 | −0.853 | prompt |
| frozen baseline, flat-lay prompts | 3.754 | −0.502 | prompt |
| per-shape remapping, countable | 2.296 | +0.082 | **diagnostic only** |
| per-shape remapping, flat-lay | 1.753 | +0.299 | **diagnostic only** |
| — measurement-side median inversion | 1.479 | +0.408 | not generative |

**No trained sampler has beaten anything.** Method C emitted 0.19 objects per image; Methods D and
B each collapsed the generator (§4). The samplers are the open question; the rows above are the
problem statement.

### Is the count-free design even reachable?

Δφ must drag x̂0 from where the count-free prompt lands to where each count lives. Measured:

| | d(x̂0) |
|---|---|
| adjacent counts (n, n+1) | 0.536 |
| full span, 1 ↔ 10 | **0.965** |
| furthest count from the count-free start | **1.002** |
| where the decode stops being a scene | **≈ 1.17** |

**It fits.** The whole count range lives inside ~1.0 of x̂0 distance — not 9 × 0.536 = 4.8, which
is what a straight line would cost — and every count is inside the safe budget. The 1.002 is an
upper bound besides, since the sampler need not land on the count-in-prompt image specifically;
any image with n objects will do and some are nearer. The margin is only 15 %, so δ_x ≈ 1.1.

Two details from the same matrix. The spacing **shrinks as n grows** (0.67 for 1→2, 0.41 for
7→8), which is the shape `g_log` in `L_ord` assumes — but log predicts a 5.3× ratio where the
measurement gives 1.6×, so the spec's metric is right in form and far too aggressive in degree.
And the count-free prompt lands **near the high counts** (0.60 to n = 10, 0.97 to n = 1): a bare
plural renders as *many*, so the sampler's work is nearly all at the low end, and that is where it
should be expected to fail first.

## 2. The frozen generator already hears the count

With countable-scene prompts (`"{count} {color} {shape} on a white background"`), fitted on
held-out seeds:

```
counted = 1.514 n + 0.069          (1.754 n - 0.658 once the counter's dense-scene failures are excluded)
```

The offset is zero. The model is not failing to hear the number — it hears it and draws about
twice too many. The median is the cleaner statement: exactly `2n − 1` for n ≤ 3 and `2n − 2` for
n = 4–6, then falling behind as generator and counter saturate.

**The value of the gain depends on the detector's operating point, and was checked by hand.** At
OWLv2 threshold 0.25 the same images fit `1.033 n + 0.546` — apparently a perfectly calibrated
generator. Counting eight images by eye settles it: at 0.25 the detector returns 0, 1 or 2 on
images that visibly hold ten or more objects, while at 0.10 it tracks the visible count closely
(11 against ~12, 9 against ~8–9, 17 against ~16–18). The flattering number at 0.25 is two errors
cancelling — a detector that misses objects, applied to a generator that draws too many.
**0.10 is the right operating point**, with the qualification that it still reads slightly high
against a manual count, so the true gain is a little below 1.514.

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

**And the budget is sufficient.** Walking a slerp path between two noises the model turns into
different counts — spherical interpolation holds ‖z‖ constant, so the path cannot collapse, and a
flat image-sd column confirms it:

| d(z, start) | median \|Δcount\| | ≥ 1 obj | ≥ 2 obj | img sd |
|---|---|---|---|---|
| 0.016 | 0 | 11 % | 11 % | 51.0 |
| 0.055 | 1 | 67 % | 33 % | 51.2 |
| **0.079** | **1** | **67 %** | **44 %** | **51.4** |
| 0.110 | 2 | 89 % | 56 % | 51.6 |
| 1.418 | 10 | 100 % | 100 % | 60.2 |

At d = 0.079, inside the 0.08 radius where a learned direction still draws a scene, the median
count change is **one object** — and one object is exactly what the stranded targets need. The
leverage available and the leverage required are the same size. (Caveats: slerp paths are a lower
bound per unit distance but stay on the Gaussian manifold, which learned directions do not; the
response is very path-dependent; and nine paths is thin. The *mean* column of that probe is
unusable — a single detector flip of 17 → 1 contributes 16/9 = 1.78 on its own.)

So the obstacle for Methods A, C and D was never an insufficient budget. It is that the direction
their losses choose destroys the image before spending it.

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

**The detector threshold changes the ceilings but not the result.** Re-counting the same images at
threshold 0.25 instead of 0.10:

| threshold | fitted gain | raw skill | remapped, per shape | affine ceiling | zero % |
|---|---|---|---|---|---|
| 0.10 | `1.514 n + 0.069` | −0.853 | **+0.082** | +0.341 | 0.7 |
| 0.25 | `1.033 n + 0.546` | −0.412 | **+0.062** | +0.062 | 9.9 |

The generation-side result holds at both (global remapping scores +0.016 and +0.080). The
*ceilings* move a lot, because at 0.25 the apparent gain is already 1 and a calibration has
nothing left to correct — what remains is spread, which no affine map touches. §2 records why
0.10 is the operating point to believe.

**Four detector defects were found and fixed**, all of which inflated the baseline: proper-noun
queries ("blue cross" returned nothing on 90 % of images), a threshold at 3× the library default,
a padded-coordinate normalisation bug, and nested-box double counting that plain IoU cannot
suppress, since a small box inside a large one has IoU = a/A.

**Degenerate runs flag themselves.** `scripts/results_table.py` leads with spread, per-shape slope,
counted mean and zero rate, and marks a run degenerate rather than letting a good MAE speak for it.
This exists because Method C improved MAE by 2.855 against the baseline it was measured on,
significant on a paired test, while producing 0.19 objects per image.

---

## 6. Where the work is now

**The spec assumes a differentiable count, and that assumption is what failed.** Measured at the
level a per-image loss actually operates on:

| method | signal | correlation with the score, per image |
|---|---|---|
| C | peak separation on cross-attention | degenerate in practice: 0.19 objects per image, 82 % empty |
| D | χ on the object token's cross-attention | **+0.154** within-n |
| B | density counter on x̂0 | **+0.186** pooled, **−0.007** within-n on the total |

Method D's layer was chosen on a reported correlation of **+0.989**, which turned out to be the
correlation of ten per-count *means*; per image it is +0.270, and χ's spread at fixed n (4.13) is
the size of its entire range across counts (4.13). Method B's counter was retrained on 634
OWLv2-labelled generator latents oversampled to 24 % of its training set, and that made it
**worse** (+0.281 → +0.186).

This is not a statement about the sampler. `probe_perturbation.py --mode leverage` showed a
displacement of 0.079, inside the measured safe radius, changes the count by a median of one
object — exactly the correction the task requires. **The obstacle is the measurement.**

**So the requirement that the measurement be differentiable is what gets dropped.**
`scripts/train_sampler_rl.py` trains the sampler by policy gradient with **OWLv2 itself as the
reward** — no gradient through the generator or the counter, so no proxy at all:

```
z = eps + w * Dphi(eps, n, c_bar) + sigma * xi        xi ~ N(0, I)
grad_phi E[R] = E[ A * (w/sigma) * <xi, grad_phi Dphi> ]
```

with a per-count EMA baseline, batch-normalised advantages, and a reward of −|counted − n| / n.
The estimator is unit-tested against a known optimum. Two constraints are **enforced rather than
priced**, which is the session's recurring lesson: the perturbation is hard-projected to
d(z, ε) ≤ 0.07, and the count is absent from the prompt so Δφ is the only carrier of n.

The first attempt collapsed in 25 steps because the reward made collapse optimal — the count-free
baseline draws 15.7 objects for a mean request of 5.5, so emitting nothing scores an absolute
error of 5.5 against 10.2 for drawing normally. Hence the projection and the relative error.

**The bar is skill 0.000**, not the count-free baseline's −3.596: a sampler that merely thins a
dense field toward the mean request would reach ≈ 0.000 while carrying no count information, so
only a result above 0.000 demonstrates that Δφ varies the count with n.

### What the policy-gradient route established

Six runs, each ended by a different real defect — the reward making collapse optimal, a penalty
that merely priced degradation, a barrier with no exit, a fixed radius in a 15 % corridor, an
advantage normalised across counts whose reward scales differ 16-fold, and finally a quality term
that **rewarded adding objects** (a correct 1-object image scores img_sd 45.4 against 63.2 for 14
objects, so a floor of 45 rejected 49 % of low-count images). Each fix was justified; none
produced learning.

**The pattern is the result.** Reward, advantage sign and the score-function estimator were each
verified independently — the estimator against a known optimum — so this is not a bug. At every
radius and under every reward shape, **the policy found directions that add objects and never
found directions that remove them.** Adding texture to a latent is easy; removing objects while
keeping a valid scene is not. With the safe corridor between "reaches n = 1" (0.97) and "destroys
the decode" (1.17) only 15 % wide, the mechanism is **reachable but not trainable by this route at
this budget**.

### Still unanswered

- **Trainability, not feasibility.** 1600 reward evaluations per run is small for policy gradient.
  A null result cannot separate "the mechanism cannot" from "the search was too small", and
  settling that needs an order of magnitude more compute than one 8 GB card gives.
- **`L_x0anc` has never been run.** It is implemented, tested and configured, and every collapse
  in this project would have been caught by it; the differentiable methods were abandoned for
  proxy failure before it was tried.
- **Why removal is hard.** That adding objects is easy and removing them is not is an observation
  across six runs, not a measurement. It would be worth testing directly — e.g. whether a slerp
  path toward a *lower*-count noise degrades the image more than one toward a higher-count noise.
