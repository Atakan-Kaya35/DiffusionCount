# Troubleshooting

## Installation and hardware

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` / `sm_120 is not compatible` | PyTorch built for an older CUDA. RTX 50xx needs CUDA ≥ 12.8 | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128` |
| `torch.cuda.is_available()` is False | driver too old, or a CPU wheel | update the NVIDIA driver (R570+), reinstall the cu128 wheel |
| `CUDA requested but not available; falling back to CPU` warning | as above | as above. Training on CPU is not practical |
| 401/403 when downloading `stabilityai/sdxl-turbo` | licence not accepted, or not logged in | accept the licence on the Hub page, then `huggingface-cli login` |
| disk full during download | Hugging Face cache on a small disk | `export HF_HOME=/path/with/space` |

## Memory

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in sampler training | add `--config configs/gpu_8gb.yaml`, or lower `train.eps_per_batch` (raise `train.grad_accum` to keep the effective batch). Keep `generator.gradient_checkpointing=true` |
| OOM only with Method A | the self-attention layer is too large: use a 16×16 or 32×32 layer, not 64×64 (`scripts/list_layers.py` shows grids) |
| OOM only with Method B, image counter | backprop through the VAE decoder. Prefer the latent counter, or lower the batch |
| OOM in pre-checks | `--set precheck.batch_size=4` or fewer `precheck.resolutions` |
| OOM with SDXL-Turbo | use `configs/sdxl_turbo.yaml` (one base noise per micro-batch). On 8 GB, SDXL-Turbo training is unlikely to fit |

## Configuration errors

| Error | Fix |
|---|---|
| `attention layers not found in U-Net` / `is not a self-attention layer` | layer names differ between SD-Turbo and SDXL-Turbo. Run `scripts/list_layers.py` and copy exact names (`attn1` = self, `attn2` = cross) |
| `object word '…' not found in tokenised prompt` | a template in `data.templates` lacks `{shape}`, or a shape word is unusual. Every template must contain `{shape}` |
| `counter has no channel for category '…'` | the counter was trained with different colours or shapes than `data.*`. Retrain the counter or align `data.colors` / shapes |
| `method.stages must contain a stage starting at step 0` | the first stage needs `start: 0` |
| `unknown terms … in stage` / `no weight for terms` | stage term names must be in the method's `TERMS` or `anc/pri/ord`, and each needs a `method.weights` entry |
| `counts_per_eps J=… exceeds n_max` | J counts are drawn without replacement from 1…N_max |

## Training behaviour

| Symptom | Likely cause and what to try |
|---|---|
| `skipped=1` often | non-finite loss or failed eigendecomposition. For A: raise `method.spectral.ridge`, raise the start of `method.spectral.T`, lower γ. Check `grad_norm` |
| A: `A/valid_frac` low | the foreground is too small: lower `method.spectral.tau_a` or `min_foreground`. Check ā at that layer (pre-check A) |
| A: `A/k_hat_mean` stuck near 1 | the whole foreground forms one cluster: raise γ, lower τ, try another layer or `affinity=feature_cosine` |
| A: `A/k_hat_mean` near K | everything counts as a component: raise τ_a, lower `tau`, or set `drop_self_loops=true` |
| B: counter numbers perfect, hard counter unchanged | counter exploitation (Remark 3): augmentation, harvested data, raise β, ensemble |
| B: `B/stray_mass` high | stray objects of other categories: raise β |
| C: `C/frac_peaks_ge_n` stays low | the cross-attention does not split into peaks (Remark 4). Check pre-check C, try other layers or less smoothing |
| `geom/d_anchor_max` well above δ | the method loss outweighs the anchor: raise `method.weights.anc` or lower the method weight |
| `geom/norm_sq` far from 1 | raise `method.weights.pri` |
| previews: sampler rows identical to baseline after many steps | weights too small or LR too low: check `loss/<method term>` is moving and the term is active in the current `stage` |
| previews: broken or noisy images | z left the noise distribution: lower `geometry.delta`, raise `anc` / `pri` |
