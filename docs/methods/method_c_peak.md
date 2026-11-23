# Method C: cross-attention peak separation (spec §5)

Code: `countsampler/novel/losses/peak.py` (losses), `countsampler/novel/methods/method_c.py` (objective).
Config: `configs/method_c.yaml`.

## Idea

Use only the object token's cross-attention map ā (head-averaged, optionally smoothed, min–max normalised). It uses no
self-attention and no counter. If the model is going to draw n objects, ā should show **n strong, separated peaks** and
nothing else. Three losses push it there:

- **L_act** keeps n peaks strong;
- **L_rep** keeps them at least r_min apart;
- **L_supp** drives any further peaks to zero.

## The repair: differentiable centres

As originally written, L_rep is a function of peak **coordinates** found by NMS or argmax. Those are piecewise constant,
so no gradient reaches ā. The spec keeps the three terms and makes the centres differentiable. This repair is new to
our version.

| Step | Code |
|---|---|
| 1. Find local maxima of ā with 3×3 max-pooling (above `peak_min_value`), **stop-gradient** | `local_maxima` |
| 2. Initialise n centres at the n largest maxima. If there are fewer than n, fill the rest by **farthest-point selection** (the "missing" objects) | `init_centers` |
| 3. Run J ≈ 3 iterations of attention-weighted soft k-means (Eq. 18) | `soft_kmeans` |
| 4. Soft peak value per cluster: a soft maximum of ā inside it | `soft_peak_values` |

```math
q_{ik} = \frac{\exp(-\lVert x_i - m_k\rVert^2/\tau_q)}{\sum_l \exp(-\lVert x_i - m_l\rVert^2/\tau_q)}, \qquad
m_k \leftarrow \frac{\sum_i q_{ik}\,\bar a_i\,x_i}{\sum_i q_{ik}\,\bar a_i} \qquad \text{(Eq. 18)}
```
```math
a_k = \frac{\sum_i q_{ik}\,\bar a_i\,e^{\bar a_i/\tau_p}}{\sum_i q_{ik}\,e^{\bar a_i/\tau_p}}
```

The centres m_k and peak values a_k are now smooth functions of ā. The losses:

```math
\mathcal L_{\text{act}} = -\frac1n \sum_{k=1}^n \log(a_k + \varepsilon) \qquad \text{(Eq. 19)}
```
```math
\mathcal L_{\text{rep}} = \frac{2}{n(n-1)} \sum_{i<j} \max\big(0,\ r_{\min} - \lVert m_i - m_j\rVert\big)^2 \quad (n \ge 2) \qquad \text{(Eq. 20)}
```
```math
\mathcal L_{\text{supp}} = \sum_{p \in \mathcal S} \bar a(p)^2, \quad
\mathcal S = \{p : p \text{ local max of } \bar a,\ \lVert x_p - m_k\rVert > r_s\ \forall k\} \qquad \text{(Eq. 21)}
```

Local maxima are located under stop-gradient, so only the values ā(p) carry gradient in L_supp, as in the original.
A cluster that gets no attention mass keeps its previous centre instead of jumping to (0, 0).

**Schedule (Eq. 22):** stage 1 {act, supp, anc, pri} → stage 2 (step 2000) adds rep → stage 3 (step 4000) adds ord.

## Units: everything is in tokens

Coordinates are token-grid positions of the attention layer (32×32 for the default layer at 512 px). The spec says r_min
and r_s should be "set from the known shape radius". In the synthetic setup, shapes have a radius of 4.5–9 % of the
image, so at a 32×32 grid:

| | tokens |
|---|---|
| object radius | 0.045 × 32 ≈ 1.4 to 0.09 × 32 ≈ 2.9 |
| centre distance of two touching objects | ≈ 2.9 – 5.8 |
| default `r_min` | 3.0 |
| default `r_s` | 2.5 |

If you change the attention layer's resolution (e.g. to 16×16), halve these.

## Knobs

| Key | Default | Notes |
|---|---|---|
| `method.attention.cross_layers` | `up_blocks.2.attentions.0.transformer_blocks.0.attn2` | set from pre-check C |
| `method.attention.smoothing_sigma` | 1.0 | smoothing of ā (tokens). Smoothing merges noise peaks |
| `method.attention.grid` | null | resample ā to a fixed grid, e.g. `[32, 32]`, when averaging layers of different resolutions |
| `method.peak.kmeans_iters` | 3 | J |
| `method.peak.tau_q` | 4.0 | assignment temperature (tokens²). Smaller means harder assignment |
| `method.peak.tau_p` | 0.1 | soft-max temperature for a_k. Smaller means closer to a hard max |
| `method.peak.r_min`, `r_s` | 3.0, 2.5 | see "Units" |
| `method.peak.peak_min_value` | 0.1 | minimum ā for a maximum to seed a centre |
| `method.peak.supp_min_value` | 0.0 | minimum ā for a maximum to count in L_supp |
| `method.peak.fps_min_value` | 0.0 | farthest-point fill only considers tokens with ā ≥ this |
| `method.peak.diag_peak_threshold` | 0.3 | diagnostic only |

## Diagnostics (log keys)

| Key | Meaning |
|---|---|
| `C/frac_peaks_ge_n` | share of samples where ā has ≥ n local maxima above `diag_peak_threshold`. This is the Remark 4 diagnostic |
| `C/num_peaks_minus_n` | mean (number of peaks − n). Positive means extra peaks, negative means missing ones |
| `C/peak_value_mean` | mean soft peak value a_k over the n centres. L_act pushes it towards 1 |

## Risks (Remark 4)

- Method C has **no count read-out and no external anchor**. n enters only as the number of centres.
- It assumes the object token's cross-attention can show separate peaks. Our earlier view of cross-attention as
  **one merged blob** contradicts that. The first thing to check is how often ā has ≥ n peaks on the frozen model:
  pre-check C (`precheck_C_summary.csv`, column `frac_peaks_ge_n`).
- L_act can strengthen a peak at a location that holds no object. Farthest-point-filled centres start in empty regions on purpose.

## How to modify

- **Different centre initialisation:** change `init_centers`, e.g. to restrict farthest-point candidates to the foreground with `fps_min_value`.
- **Different separation term:** edit the L_rep block in `peak_losses` (it uses a NaN-safe distance).
- **Add a term:** return it from `peak_losses`, add its name to `TERMS` in `method_c.py`, and add a weight and stage in the config.
