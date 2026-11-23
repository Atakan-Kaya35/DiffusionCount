# Method A: spectral instance count from self-attention (spec §3)

Code: `countsampler/novel/losses/spectral.py` (losses), `countsampler/novel/methods/method_a.py` (objective).
Config: `configs/method_a.yaml`.

## Idea

The object token's **cross-attention** says *where* the object is (the foreground). **Self-attention** among the
foreground pixels says *how it is grouped*: a pixel mostly attends to pixels of its own object, so separate instances
form separate clusters. Build a graph on the foreground tokens from self-attention, and count its connected
components with spectral graph theory:

> If a graph has exactly r connected components, the eigenvalue 0 of its normalised Laplacian has multiplicity r.

So **the number of near-zero eigenvalues counts the instances**. A smooth version of that count is pushed towards the
prompted n. No external counter or detector is used. The prompt count n is the only label.

## Pipeline (one unrolled step)

| Step | Spec | Code |
|---|---|---|
| 1. Read ā: object token's cross-attention, head-averaged, resized to the self-attention grid, lightly smoothed, min–max normalised | §3.1 | `generator.attention.object_attention_map` |
| 2. Read S: head-averaged self-attention probabilities, computed explicitly (rows sum to 1) | §3.1 | `generator.attention.CaptureAttnProcessor`, `mean_self_attention` |
| 3. Foreground F: tokens with ā ≥ τ_a, top K by ā, **under stop-gradient** | §3.1 | `select_foreground` |
| 4. Affinity on F | Eq. 10 | `attention_affinity` (or `feature_affinity`) |
| 5. Normalised Laplacian with degree ridge | Eq. 11 | `normalized_laplacian` |
| 6. Eigenvalues in float64, batched | §3.3 | `torch.linalg.eigvalsh` |
| 7. Soft count, count loss, gap loss | Eq. 12–14 | `soft_count`, `count_loss`, `gap_loss` |

```math
W_{ij} = \Big(\tfrac12 (S_{ij} + S_{ji})\Big)^{\gamma}, \quad i,j \in \mathfrak F \qquad \text{(Eq. 10)}
```
```math
L = I - D^{-1/2} W D^{-1/2}, \quad D = \operatorname{diag}(W\mathbf 1 + \rho) \qquad \text{(Eq. 11)}
```
```math
\hat k = \sum_{i=1}^{K} \sigma\!\Big(\frac{\tau - \lambda_i}{T}\Big), \qquad
\mathcal L_{\text{cnt}} = \operatorname{Huber}_\eta\big(\log(\hat k + \varepsilon_k) - \log n\big), \qquad
\mathcal L_{\text{gap}} = \max(0,\ \tau_g - \lambda_{n+1}) \qquad \text{(Eq. 12–14)}
```

- **L_cnt** handles too few and too many groups, on a log scale (it matches g_log).
- **L_gap** sharpens the too-many case. With more than n groups, λ_{n+1} ≈ 0 and the term fires. With fewer, λ_{n+1} is already large and the term is inactive.

Tokens of the top-K that fall below τ_a are **masked**: their rows and columns of W are zeroed, which makes them isolated
nodes with eigenvalue exactly 1. They never count as groups. Samples with fewer than
`max(min_foreground, n+1)` foreground tokens are excluded from the loss (K > n is required, §3.2).

## Why the gradient is stable (§3.3)

Both losses depend on **eigenvalues only**. The gradient of an eigenvalue with respect to L is u_i u_iᵀ, with no
1/(λ_i − λ_j) term, so PyTorch's ordinary `eigvalsh` backward is exact and needs no custom backward.
The code computes in float64 on K×K matrices (K ≤ 256), uses a degree ridge, clips gradients, and anneals γ and T:
γ goes 1 → 2 and T goes 0.05 → 0.02 over the first 3000 steps by default. Foreground selection is under stop-gradient.

## Schedule (Eq. 15)

```math
\mathcal L_A = \lambda_{\text{cnt}}\mathcal L_{\text{cnt}} + \lambda_{\text{gap}}\mathcal L_{\text{gap}} + \lambda_{\text{anc}}\mathcal L_{\text{anc}} + \lambda_{\text{pri}}\mathcal L_{\text{pri}} \;[+\ \lambda_{\text{ord}}\mathcal L_{\text{ord}}]
```

Stage 1 {cnt, anc, pri} → stage 2 (step 2000) adds gap → stage 3 (step 4000) adds ord, run once with g_log and once with g_lin.

## Knobs

All open choices in §9 are set by the offline scan ([workflow.md](../workflow.md#1-pre-checks-and-offline-scan-before-any-training)).

| Key | Default | What it does / when to change |
|---|---|---|
| `method.attention.self_layers` | `up_blocks.2.attentions.0.transformer_blocks.0.attn1` (32×32) | the layer ℓ for S. Several same-resolution layers are averaged |
| `method.attention.cross_layers` | the matching `attn2` | defines F only. Several layers are resized and averaged |
| `method.attention.smoothing_sigma` | 0.5 | Gaussian blur of ā in token units |
| `method.spectral.affinity` | attention | `feature_cosine` = Remark 2 fallback: W = max(0, cos(f_i, f_j))^γ on the self-attention output features |
| `method.spectral.K` | 256 | foreground cap. Memory is tiny, but n must stay below the foreground size |
| `method.spectral.tau_a` | 0.3 | ā threshold. Lower it if `A/valid_frac` or `A/n_fg_mean` is low |
| `method.spectral.gamma` | 1 → 2 | sharpening. Higher means more contrast between within- and between-instance weights |
| `method.spectral.T` | 0.05 → 0.02 | soft-count temperature. Lower is sharper but gives a spikier gradient |
| `method.spectral.tau` | 0.1 | what counts as a "near-zero" eigenvalue |
| `method.spectral.tau_g` | 0.2 | gap threshold for λ_{n+1} |
| `method.spectral.ridge`, `ridge_mode` | 1e-3, relative | ρ = ridge × mean foreground degree (stop-grad). See [spec_decisions.md](../spec_decisions.md) |
| `method.spectral.drop_self_loops` | false | the spec keeps W_ii. `true` stops weakly connected single tokens from counting as groups |
| `method.spectral.eps_k`, `huber_eta` | 0.1, 0.5 | ε_k and η in Eq. 13 |
| `method.spectral.min_foreground` | 8 | minimum |F| for a sample to count |

## Diagnostics (log keys)

| Key | Meaning |
|---|---|
| `A/k_hat_mean` | mean soft count k̂ |
| `A/k_hat_abs_err` | mean \|k̂ − n\| over valid samples (n = prompted count, the training target) |
| `A/lambda_next_mean` | mean λ_{n+1}. It should rise above τ_g as L_gap does its job |
| `A/n_fg_mean` | mean foreground size \|F\| |
| `A/valid_frac` | share of samples that entered the loss |
| `A/gamma`, `A/T` | current annealed values |

With multi-step unrolls, the keys get a `@t<timestep>` suffix.

## Risks (Remark 2)

The signal comes from the network that generates the image, so it can be **consistent and wrong**: touching objects
read as one group, or one object fragmented into several. The prompt count anchors the loss but does not check the image.
Only the hard-counter evaluation shows real progress. Pre-check A tests exactly this on identical and touching
synthetic shapes.

## How to modify

**Add a loss term** (e.g. an eigengap ratio):

1. Write the function in `spectral.py` and return its value in `SpectralResult` (or compute it from `res.eigvals`).
2. In `method_a.py`, add the name to `TERMS = ("cnt", "gap", "ratio")` and return it from `method_terms`.
3. In `configs/method_a.yaml`, add `ratio` to `method.weights` and to the stages where it should be active.

**Try a different affinity:** add a function next to `attention_affinity` and a branch on `method.spectral.affinity`
in `method_a.py`. The Laplacian and loss code do not change.

**Read attention at several layers:** list them in `method.attention.self_layers`. They must share a resolution
(`scripts/list_layers.py` shows the grids).
