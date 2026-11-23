# Spec → code map

Every section and equation of [`count_sampler_formalization_v5.pdf`](../count_sampler_formalization_v5.pdf), and where it lives.
Paths are relative to the repository root. Files under `countsampler/novel/` carry a ★ NOVEL ★ banner.

## §1 Setting and notation

| Spec | Code |
|---|---|
| Frozen G: text encoder, U-Net, VAE | `countsampler/generator/turbo.py` `FrozenTurboGenerator` (`freeze`) |
| Prompt p_n, e.g. "n red circles" | `countsampler/data/prompts.py` `build_prompt`, `PromptSampler` |
| c(p_n), pooled c̄ | `FrozenTurboGenerator.encode_prompts` (`PromptEncoding.hidden`, `.pooled`) |
| Attention read at layer ℓ, timestep t, first step(s) | `countsampler/generator/attention.py` `CaptureAttnProcessor`; `generator.timesteps` |
| Eq. 1, clean estimate x̂0, ᾱ_t clamped, graph kept | `FrozenTurboGenerator.forward_at`; `generator.min_alpha_bar` |
| One U-Net pass gives attention maps and ε̂ | `FrozenTurboGenerator.unroll(..., capture=True)` returns `StepOutput` with both |

## §2 Shared components

| Spec | Code |
|---|---|
| Eq. 2, sampler, FiLM on sin-enc(log n) and c̄, zero-init last layer, mixing weight w | `countsampler/novel/shared.py` `CountNoiseSampler`, `sinusoidal_count_encoding`, `FiLMResBlock` |
| Eq. 3, normalised distance | `normalized_distance` |
| Eq. 4, g_log, g_lin | `g_log`, `g_lin` |
| Eq. 5, L_ord regression | `GeometryLosses.ordering_loss` |
| Eq. 6, L_ord ranking | `GeometryLosses.ranking_loss` |
| Eq. 7, L_anc | `GeometryLosses.anchor_loss` |
| Eq. 8, α = 2δ/g_max | `alpha_scale`, `g_max` |
| Eq. 9, L_pri | `GeometryLosses.prior_loss` |
| Remark 1 (L_ord alone is degenerate) | stage schedules in `configs/method_*.yaml`; `tests/test_shared.py` |

## §3 Method A

| Spec | Code |
|---|---|
| §3.1 ā (head-averaged, min–max, smoothed) | `countsampler/generator/attention.py` `object_attention_map`, `minmax_normalize` |
| §3.1 explicit S = softmax(QKᵀ/√d) at a mid-resolution layer | `CaptureAttnProcessor`, `mean_self_attention` |
| §3.1 foreground F, τ_a, top-K, stop-gradient | `countsampler/novel/losses/spectral.py` `select_foreground` |
| Eq. 10, W = (½(S+Sᵀ))^γ | `attention_affinity`, `gather_submatrix` |
| Eq. 11, normalised Laplacian, ridge ρ | `normalized_laplacian` |
| Eq. 12, soft count k̂ | `soft_count` |
| Eq. 13, L_cnt (Huber on log ratio) | `count_loss`, `huber` |
| Eq. 14, L_gap | `gap_loss` |
| §3.3 float64 batched eigendecomposition | `spectral_losses_from_affinity` (`torch.linalg.eigvalsh`) |
| §3.3 ridge, clipping, γ/T annealing | `normalized_laplacian`; `train.grad_clip`; `countsampler/novel/methods/method_a.py` `params` |
| §3.3 stop-gradient on F | `select_foreground` (decorated `no_grad`) |
| Eq. 15 + schedule | `countsampler/novel/methods/method_a.py`; `configs/method_a.yaml` |
| Remark 2 fallback: cosine of self-attention features | `feature_affinity`; `method.spectral.affinity=feature_cosine` |

## §4 Method B

| Spec | Code |
|---|---|
| §4.1 C_ψ, K density channels, soft count | `countsampler/novel/components/counter.py` `DensityCounter` |
| §4.1 Gaussian targets normalised to 1 | `countsampler/data/shapes.py` `density_target` |
| §4.1 loss: MSE with scale s + L1 on counts | `countsampler/training/counter_trainer.py` `train_counter` |
| §4.1 heavy augmentation | `countsampler/data/counter_data.py` `augment_batch` |
| §4.1 x̂0 from the frozen generator as extra data | `countsampler/data/harvest.py`, `scripts/harvest_generator_samples.py` |
| §4.1 softplus head bias strongly negative | `DensityCounter.__init__` (`head_bias_init`) |
| §4.1 validation: MAE, exact accuracy, held-out combinations | `evaluate_counter`, `heldout_combos` |
| §4.1 frozen: eval, requires_grad=False, not in optimiser, not no_grad | `freeze_counter`, `load_counter`; `countsampler/novel/methods/method_b.py` |
| §4.1 counter on latents (preferred) or decoded images | `counter.input_space`; `method_b.counter_input` |
| Eq. 16, L_den, n_k = 0 for unmentioned categories | `countsampler/novel/losses/density.py` `density_count_loss`, `category_targets` |
| §4.2 tolerance / Huber variants, w(t) | `density_count_loss(variant=...)`, `timestep_weight` |
| Eq. 17 + schedule | `countsampler/novel/methods/method_b.py`; `configs/method_b.yaml` |
| Remark 3, independent hard counter | `countsampler/evaluation/hard_counters.py` |

## §5 Method C

| Spec | Code |
|---|---|
| ā of the object token only | `countsampler/novel/methods/method_c.py` `method_terms` |
| Centre init: n largest local maxima (3×3), stop-gradient, farthest-point fill | `countsampler/novel/losses/peak.py` `local_maxima`, `init_centers` |
| Eq. 18, soft k-means | `soft_kmeans` |
| Soft peak value a_k | `soft_peak_values` |
| Eq. 19–21, L_act, L_rep, L_supp | `peak_losses` |
| Eq. 22 + schedule | `countsampler/novel/methods/method_c.py`; `configs/method_c.yaml` |
| Remark 4 diagnostic: NMS peaks above a threshold vs n | `PeakResult.num_peaks`; log key `C/frac_peaks_ge_n`; pre-check C |

## §6–§8

| Spec | Code |
|---|---|
| §6 methods run separately | one config per method; `countsampler/novel/methods/__init__.py` `build_method` |
| §7 synthetic shapes with held-out shapes | `data.train_shapes`, `data.heldout_shapes`; `countsampler/data/shapes.py` |
| §7 steps 1–6 | `countsampler/training/sampler_trainer.py` `SamplerTrainer.train_step` |
| §7 step 3 gradient checkpointing | `generator.gradient_checkpointing` |
| §7 step 6 clipping and skip-on-NaN | `SamplerTrainer.train_step` |
| §7 ablation without the number | `data.include_count=false` |
| §8 count accuracy with an external counter | `countsampler/evaluation/evaluate.py` `run_evaluation` |
| §8 CoCoCount and D2D benchmarks | `countsampler/data/benchmarks.py`, `countsampler/evaluation/benchmark_eval.py`, `scripts/prepare_benchmarks.py`, `scripts/evaluate.py --benchmark` |
| §8 one shared counter (OWLv2 / YOLOv9; CountGD external) | `countsampler/evaluation/hard_counters.py` |
| §8 comparison table | `scripts/compare.py` |
| §8 per-seed count histograms | `per_seed_hist.csv`, `seed_spearman_mean` |
| §8 diversity and image quality, grid-layout detection | `countsampler/evaluation/metrics.py` `pairwise_diversity`, `nn_regularity`, `ClipScorer` |
| §8 method-specific pre-checks | `countsampler/evaluation/prechecks.py`, `scripts/precheck.py` |

## §9 Open choices

Every open choice is a config key marked `OPEN` in the YAML files. See [configuration.md](configuration.md) for the full list and
[spec_decisions.md](spec_decisions.md) for the choices the code makes.
