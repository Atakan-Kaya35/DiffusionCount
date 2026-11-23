# Spec decisions and gaps

Where the spec is open or ambiguous, the code makes a choice. Each choice is listed here with its config switch.
This page also lists what the spec mentions that is **not implemented yet**. If you change a default, update this page.

## Interpretation choices

### First U-Net input (`generator.input_scaling`)

§1 says "for a noise-prediction model with noisy input x_t = z_n". SD-Turbo's actual diffusers pipeline (Euler, trailing
timesteps) scales the initial latent by σ_max and then divides it by √(σ²+1), so the U-Net sees √(1−ᾱ_T)·z, which is
0.9977·z at T = 999. Eq. 1 is applied to whatever x_t is in both cases.

- `pipeline` (default): x_T = √(1−ᾱ_T)·z_n. The frozen baseline (z = ε) then reproduces official SD-Turbo outputs exactly. This is tested against diffusers for 1, 2 and 4 steps.
- `identity`: x_T = z_n, the literal reading.

The difference is a 0.23 % input scale. We default to `pipeline` so that "frozen baseline" means the official model.

### Multi-step unrolls

For `generator.timesteps` with more than one step, x_{t'} = √ᾱ_{t'} x̂0 + √(1−ᾱ_{t'}) ε̂. This equals SD-Turbo's
deterministic Euler step exactly. SDXL-Turbo ships an *ancestral* Euler scheduler, whose added noise we leave out (deterministic,
and differentiable without sampling noise). Per §9, the method loss is averaged over the unrolled steps (`method.loss_steps`).

### Which count pairs the geometry losses use (`geometry.over`)

Eq. 5 sums over all 1 ≤ n < m ≤ N_max. §7 step 5 computes the losses "across the J noises of each ε". The default is
`sampled` (§7). `all_counts` evaluates the sampler for every n (cheap, no generator) and applies Eq. 5 literally.

### Degree ridge ρ (`method.spectral.ridge_mode`)

Attention probabilities are tiny (rows sum to 1 over ~1000 tokens) and are raised to γ, so a fixed ρ = 1e-3 could exceed
every degree. `relative` (default) sets ρ = ridge × mean foreground degree, under stop-gradient. This is equivalent to rescaling W,
which the spec notes the normalised Laplacian is otherwise invariant to. `absolute` uses ρ as given.

### Self-loops (`method.spectral.drop_self_loops`)

Eq. 10 defines W_ij for all i, j ∈ F, including i = j. The default keeps the diagonal. With self-loops, a weakly connected
single token can count as its own component. Dropping the diagonal makes it an isolated node with eigenvalue 1 that never counts.

### Foreground masking

F has a variable size, but batched `eigvalsh` needs a fixed K. The code always takes the top-K tokens by ā and masks those below
τ_a, zeroing their rows and columns, which gives eigenvalue exactly 1, never counted. Samples with
|F| < max(`min_foreground`, n + 1) are excluded from L_cnt and L_gap (§3.2 requires K > n).

### Cosine-of-features fallback (Remark 2)

"Self-attention features" is taken to mean the self-attention output (heads concatenated, before `to_out`). With several
layers, each layer's features are L2-normalised and concatenated, so the cosine is the mean per-layer cosine.

### Counter loss "MSE … scaled by a constant s" (§4.1)

Implemented as MSE(s·D, s·target), with both sides scaled. Soft counts therefore still sum to n, consistent with the L1 count term
and with N̂_k = Σ D_k in Eq. 16. Scaling only the target would make the counter predict s·n.

### Counter categories

(colour, shape) pairs over all colours and all shapes, held-out shapes included. The counter must recognise held-out shapes for
Method B's targets. The sampler never trains on them.

### Farthest-point fill (Method C)

Missing centres are chosen among all tokens with ā ≥ `fps_min_value` (default 0, i.e. all tokens), maximising the
minimum distance to the existing centres.

### Pooled embedding c̄

SD-Turbo: CLIP `pooler_output` (EOS token). SDXL-Turbo: the projected `text_embeds` of text_encoder_2 (the same vector SDXL uses
for its own conditioning).

### Mixed precision

The frozen model runs in bf16 (`generator.dtype`). The sampler, attention probabilities, x̂0, losses and eigendecompositions
run in float32/float64. bf16 avoids the gradient underflow that fp16 can cause when backpropagating through the U-Net.

## Not implemented yet / external

| Spec item | Status | How to add it |
|---|---|---|
| §8 evaluation on **CoCoCount** and the **D2D benchmarks** | missing. Evaluation runs only on the synthetic "n colour shape" prompts | a prompt-file evaluation mode (prompt, object noun, count) with a detector counter (`owlv2`, or CountGD) |
| §8 **CountGD** as the evaluation counter | external, not pip-installable | wrap it in a class with `count(image, color, shape)` in `countsampler/evaluation/hard_counters.py` |
| §6, §8 **D2D** baseline | external | run D2D's code, then evaluate its images with the same hard counter |
| §8 image quality beyond CLIP score (e.g. FID) | missing | add to `countsampler/evaluation/metrics.py` |
| Remark 3 **ensemble of counters** | missing | load several counters in `DensityCountMethod` and average their counts |
| §4.1 pretrained counter for real images | missing | any module with `count(x) -> [B, K]` can be passed to `DensityCountMethod` |
| Combining methods | later step, as §6 says | subclass `CountMethod` and sum the other methods' terms |
