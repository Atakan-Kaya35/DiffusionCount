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

**The thesis has a positive demonstration.** With the count absent from the prompt - every target
gets the identical text - a learned Delta-phi places the object count on the requested value
better than the best constant predictor, on every cell tested. Ten cells, eight seeds each, the
seeds used neither in training nor in choosing any hyper-parameter:

```
             cell   base   Dphi   skill   corr  slope  improve          95% CI       p   div
       blue/cross   6.88   2.44  +0.512  +0.97   0.84    +4.44 [+3.26,+5.63]  0.0000  0.46
      green/cross   6.58   2.56  +0.487  +0.98   0.84    +4.01 [+2.95,+5.11]  0.0000  0.46
     yellow/cross   8.22   2.56  +0.487  +0.98   0.94    +5.66 [+4.46,+6.94]  0.0000  0.47
         red/star   7.38   2.84  +0.432  +0.98   0.69    +4.54 [+3.38,+5.75]  0.0000  0.46
      red/diamond   6.40   3.00  +0.400  +0.89   0.82    +3.40 [+2.31,+4.56]  0.0000  0.43
       green/star   9.20   3.01  +0.397  +0.91   0.64    +6.19 [+4.79,+7.62]  0.0000  0.42
        blue/star   9.53   3.02  +0.395  +0.90   0.79    +6.50 [+4.72,+8.25]  0.0000  0.44
        red/cross   6.83   3.25  +0.350  +0.96   0.80    +3.58 [+2.39,+4.74]  0.0000  0.46
     blue/diamond   6.60   3.35  +0.330  +0.94   0.77    +3.25 [+2.01,+4.53]  0.0000  0.49
     red/triangle   6.50   4.70  +0.060  +0.87   1.00    +1.80 [+0.50,+3.09]  0.0037  0.40

           POOLED   7.41   3.07  +0.385  +1.00   0.81    +4.34 [+3.92,+4.76]  0.0000  0.45
```

**Ten of ten cells beat the best constant predictor, and ten of ten beat untouched noise at
p < 0.05.** Pooled skill **+0.385** at correlation **+1.00** and slope 0.81, with the degeneracy
guard fully clean: ten distinct rounded per-target means and nine of nine monotone steps. The
baseline row is flat by construction - identical prompt, untouched noise - so the whole of the
difference is carried by Delta-phi. Cost per cell: 2.7 minutes of training plus one 96-noise
counting pass.

### What it is measured against

Every comparison here used to be against untouched noise under a count-free prompt. That arm
cannot know n - the prompt is identical for every target - so its response is flat by
construction, and beating it only shows the sampler is better than ignoring the request. The
comparison a reader wants is against the obvious alternative: **put the number in the prompt and
count what comes out.** `runs/baselines/store.json` holds that arm, keyed by (colour, shape,
count, seed, mode), built once and read by every scoring script rather than regenerated.

```
             cell  prompt    +cal  countfree  constant  dphi.75  dphi.85
        red/cross    9.75    6.67       6.83      5.00     4.31     3.25
       blue/cross   10.55    6.93       6.88      5.00     3.56     2.44
      green/cross   10.53    7.48       6.58      5.00     3.55     2.56
     yellow/cross   18.70    3.83       8.22      5.00     3.79     2.56
         red/star    9.30    6.75       7.38      5.00     3.55     2.84
        blue/star    9.12   10.61       9.53      5.00     4.12     3.02
       green/star    8.39    9.20       9.20      5.00     4.92     3.01
      red/diamond    7.29    8.79       6.40      5.00     3.71     3.00
     blue/diamond    6.33   20.18       6.60      5.00     4.51     3.35
     red/triangle   11.95    5.66       6.50      5.00     4.86     4.70

           POOLED   10.19    8.61       7.41      5.00     4.09     3.07
```

**Asking the model for the number is the worst arm in the table, at 10.19.** Not because it
ignores the request - it tracks it at correlation +0.96 and slope 0.92 - but because it overdraws
by a near-constant margin: mean(counted - requested) = **+7.72**. Ask for 10, get 18; ask for 28,
get 35.

Since that error is almost all bias, `+cal` gives the arm its best possible form: each cell's own
fitted response curve inverted, so the user asks for whatever number actually yields what they
want. That is an oracle - it needs the curve measured in advance - and it is exactly the "ask for
a different number" tactic that this project rejects as a direction. It is included only so the
sampler faces the strongest honest version of the comparison. It reaches 8.61 pooled, still
behind. (blue/diamond's 20.18 is the inversion exploding on a slope of 0.24: a near-flat response
cannot be inverted.)

Against the prompt baseline, paired over (cell, seed, count), n = 800:

```
  improvement +7.12 objects   95% CI [+6.56, +7.69]   p = 0.0000

  requested     10   12   14   16   18   20   22   24   26   28
  prompt        18   20   21   24   26   31   31   29   31   35
  dphi@0.85     11   13   15   17   19   20   22   23   24   26
```

One caveat belongs with this. The band 10-28 is where SD-Turbo overdraws, so part of the prompt
baseline's handicap is the choice of band rather than a defect of prompting. The store makes that
testable on a different band without regenerating anything already measured.

### What it costs, stated plainly

`div` is across-seed image diversity (pixel sd at a fixed target) as a fraction of what untouched
noise has. At the operating point it is **0.45**: the sampler pulls each seed 85% of the way
toward a reference, so the images for one target resemble each other far more than raw samples do.
Pushing the blend to 0.95 scores better still - red/cross reaches +0.778 - at a diversity of 0.16,
which is close to rendering the same ten pictures for every seed. That is the degenerate solution
and it is not claimed here. 0.85 is used for every cell, fitted on none.

### The blend is the mechanism, and why

The reference noises are near-exact: gen_genpool.py fits them to within a count on every cell, and
red/triangle's land on [10,12,...,28] precisely. Delta-phi copies its training target almost
exactly (5.26 against the target's own 5.21 on red/triangle). Neither end is the bottleneck. What
fails is the step between: **a reference's count does not fully survive being blended with an
unseen seed**, and the fraction that survives is the blend. Raising it from 0.75 to 0.85 lifted
pooled skill from +0.182 to +0.385 and turned every remaining negative cell positive.

The error it removes is scatter, not bias. On the cells that were failing, mean(counted - target)
measured -0.24 and -0.75 - near zero. An earlier reading of these cells as "landing too high" was
wrong, and the intercept correction it implied would have fixed nothing.

### Per-cell skill is not stable across seed banks; the pooled estimate is

Scored on bank 90210 at blend 0.75, red/cross reads +0.360 and seven of ten cells are positive. On
bank 24680, identical procedure, red/cross reads **+0.138** and ten of ten are positive. The
pooled number barely moves (+0.170 against +0.182). Eight seeds is too small a bank to rank cells
against each other, so the headline is the pooled figure and the per-cell column is read as a
spread, not a league table. The earlier framing of red/cross as "the best cell" does not survive
this check.

### The architectural finding that made it possible

Everything before this was constrained to fail, and the reason is one measurement:

```
   blend  ||z-eps||/sqrt(D)  ||z||/sqrt(D)
    0.75              1.112          1.000      <- where count control lives
```

The displacement that controls the count is **1.112**. Learned *additive* directions collapse by
**0.08**, and every run here was held to a budget of 0.05–0.07. The working region is 14× further
out than the sampler was ever allowed to go.

The distance was never the problem. slerp travels to 1.42 without collapsing because it holds
‖z‖ fixed; additive `w·Δφ` at the same distance has left the shell the U-Net was trained on. So
`L_anc`, the anchor hinge, was not a safeguard — **it enforced a budget that guaranteed the
sampler could never reach the region it was meant to protect.** Every differentiable-proxy run
and every policy-gradient run was searching a ball that does not contain the answer.

The fix is one projection: `norm_preserve` rescales z back to ‖ε‖ after the additive step, so an
arbitrarily large w lands on the shell instead of off it.

### How the sampler was trained

Not from a reward. Six policy-gradient runs could not get a usable gradient from the detector.
Instead, a per-cell pool of 128 noises establishes, for each (cell, target), a noise the frozen
model demonstrably renders as that count — verified by two independent counters. That turns the
problem into supervised regression: Δφ reproduces `slerp(ε, r_{n,c}, 0.75)`. The generator is not
in the loop, so it trains in 2.6 minutes and reaches direction cosine **+0.997**.

Scored, the learned sampler (+0.272) nearly matches the selection scheme it distilled (+0.305),
so the distillation is faithful rather than lossy.

### Scope, stated plainly

- Three cells, all **diamonds**. OWLv2 cannot see pentagons in these renders — 64 of 240 images
  read as 0–2 objects while the colour counter finds a mean of 22.
- Generalises **across seeds, not across cells**. A reference fitted on one colour/shape does not
  transfer: one fixed noise spans 28.3 objects across six cells, and the cell explains 55.9 of
  the variance in the counted number against 21.9 for the requested target.
- Band **10–28**, because the count-free flat-lay prompt never renders fewer than 8 at any noise.
  The bar is therefore **5.00, not 2.500**, and this skill number is *not* comparable to the
  2.500-bar numbers elsewhere in this file.
- Slope 0.58 — it tracks at a little over half strength.
- Selection from a pool defined the target; the sampler is learned, the pool is not.

### What is closed

Three routes were run to a measurement and are done: differentiable proxies (best per-image
+0.154), policy gradient on a detector reward (six runs; no count-reducing direction exists under
a count-free prompt), and pooled — as opposed to per-cell — reference selection (negative at
every blend). The reason all three failed is now understood and is the budget finding above.

### Method notes that cost real time

- **A single metric cannot validate a search that optimises that same metric.** A reference pool
  built on the plain-plural prompt reported an oracle bound of +0.747; an independent colour
  counter found 92 objects in images OWLv2 scored as 0. The search had selected for the
  detector's blind spot. Every evaluation since counts twice and reports the agreement, which is
  how the pentagon blindness above was caught before it reached a result.
- **A per-count-mean correlation is not per-image skill.** It has misled this project three
  times: Method D read +0.989 against a per-image +0.270, and the pooled-reference run would read
  +0.36 by means against a true −0.247.
- **The prompt sets the reachable band; the noise only modulates within it.** The count-free
  flat-lay prompt spans 8–31, so targets 1–8 do not exist at any noise. That is why the RL runs
  never found a count-reducing direction — there was none to find.

### Still unanswered

- ~~Whether the mapping generalises to cells it was not fitted on.~~ **Answered: it does not.**
  Trained on ten cells with orange and purple held out entirely, Δφ is *worse than untouched
  noise* on the unseen colours (−1.18, 95% CI [−2.34, −0.02]) while still gaining on the seen
  ones (+1.57, p < 0.0001). It is not a conditioning failure: Δφ emits genuinely cell-specific
  directions (+0.30 between cells) and lands on each cell's own target at fidelity **+0.97**.
  Rebuilding the condition to make the colours more separable (pooled cosine +0.848 → token
  cosine +0.518) changed the fidelity by 0.01 and made transfer *worse*. There is simply no
  smooth structure over cells to interpolate. **Δφ must be fitted per cell** — about 96
  generations, 40 s of GPU time — after which it works. It is a per-cell calibration procedure,
  not a general-purpose count-conditioned sampler.
- Whether the band can be moved down to 1–10. That is a property of the prompt, not the sampler,
  and no noise reaches below 8 under the current one.
- Whether `norm_preserve` plus a reward now works, given that the budget that blocked the earlier
  policy-gradient runs is gone.
- `L_x0anc` remains implemented, tested, configured and never run.
