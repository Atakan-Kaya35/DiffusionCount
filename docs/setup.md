# Setup

## Hardware and drivers

| | Recommended | Minimum |
|---|---|---|
| GPU | RTX 5070 Ti (16 GB) | RTX 5060 (8 GB) with `configs/gpu_8gb.yaml` |
| Driver | a recent NVIDIA driver supporting CUDA 12.8 (R570 or newer; Blackwell cards need it anyway) | |
| Disk | ~10 GB: SD-Turbo weights (~2.6 GB), counter latent cache (~0.7 GB), checkpoints and logs | |
| Python | 3.10 – 3.12 | 3.10 |

RTX 50-series GPUs are Blackwell (`sm_120`). They need a PyTorch build compiled against **CUDA 12.8 or newer**.
Older wheels fail with `no kernel image is available for execution on the device`.

## Install

From the repository root:

```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[wandb]"` for Weights & Biases logging (then set `experiment.wandb=true`).

## Check the install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see a `+cu128` (or newer) version, `True`, and the card's name.

```bash
pytest -q
```

All tests run on the CPU against tiny random models, so they pass on any machine. They take about 10 seconds.

## Model weights

Weights download from the Hugging Face Hub on first use and are cached under `~/.cache/huggingface`
(set `HF_HOME` to move the cache, e.g. to a larger disk).

- **SD-Turbo** (`stabilityai/sd-turbo`, the default) downloads without logging in.
- **SDXL-Turbo** (`stabilityai/sdxl-turbo`) requires accepting the licence on its Hub page and logging in:

```bash
huggingface-cli login
```

The first command that loads the generator triggers the download. A good one to start with:

```bash
python scripts/list_layers.py --config configs/base.yaml
```

It prints every attention layer with its type and token grid. You need these names for `method.attention.*`.

## First GPU smoke run

Before any long run, check memory and speed with a 20-step run:

```bash
python scripts/train_sampler.py --config configs/method_a.yaml --set train.steps=20 train.preview.every=10 experiment.name=smoke_a
```

Then look at `runs/smoke_a/log.jsonl`:
- `gpu_mem_gb`: peak memory. If it is close to your card's limit, use `configs/gpu_8gb.yaml` or lower `train.eps_per_batch`.
- `sec_per_step`: multiply by `train.steps` to estimate the full run.
- `skipped`: should be 0.

Also check `runs/smoke_a/previews/`. At step 0 the baseline and sampler rows are identical, because the sampler starts as the identity.

## Development without a GPU

All tests use tiny random models and need no GPU or downloads. A CPU-only PyTorch is enough to develop and test:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Real training and evaluation need the GPU machine. See [troubleshooting.md](troubleshooting.md) if anything fails.
