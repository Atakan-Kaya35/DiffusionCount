# Count-Conditioned Noise Sampler (CS485, Bilkent)

Implementation of **"Count-Conditioned Noise Sampler: Three Methodologies" (draft v5)**,
[`count_sampler_formalization_v5.pdf`](count_sampler_formalization_v5.pdf).

A small network Δφ changes the initial noise of a **frozen** few-step text-to-image model
(SD-Turbo / SDXL-Turbo) so that the image contains the requested number of objects:

```
z_n = ε + w · Δφ(ε, n, c̄)          (Eq. 2. Only φ is trained.)
```

The sampler is the same in all experiments. Only the error signal that trains it changes:

| Method | Error signal | Extra module | Guide |
|---|---|---|---|
| **A** spectral | eigengap soft count on the self-attention graph of the cross-attention foreground | none | [docs/methods/method_a_spectral.md](docs/methods/method_a_spectral.md) |
| **B** density | frozen density-map counter C_ψ on the one-step clean estimate x̂0 | counter C_ψ | [docs/methods/method_b_density.md](docs/methods/method_b_density.md) |
| **C** peak | separated cross-attention peaks of the object token (soft centres) | none | [docs/methods/method_c_peak.md](docs/methods/method_c_peak.md) |

All three share the sampler and its geometry regularizers ([docs/methods/shared.md](docs/methods/shared.md)).

## Results

**[docs/RESULTS.md](docs/RESULTS.md) is the results package** — the headline
numbers, the figures, the baselines they are measured against, the costs, and the routes that
failed. With the count removed from the prompt entirely, a per-cell calibrated Δφ reaches pooled
skill **+0.385** over the best constant predictor across ten cells (10/10 cells positive,
p < 0.05), against **+7.12 objects** better than simply putting the number in the prompt.

## Documentation

**Start at [docs/README.md](docs/README.md).** Main pages:

- [Overview](docs/overview.md): the idea, the pipeline, how gradients reach the sampler
- [Setup](docs/setup.md): install on RTX 50-series, first smoke run
- [Workflow](docs/workflow.md): pre-checks → counter → training → evaluation → comparison
- [Benchmarks](docs/benchmarks.md): CoCoCount and D2D-style evaluation with a detector counter
- [Spec → code map](docs/spec_map.md) · [Configuration reference](docs/configuration.md) · [Outputs and logs](docs/outputs.md)
- [Architecture](docs/architecture.md) · [Spec decisions and gaps](docs/spec_decisions.md) · [Troubleshooting](docs/troubleshooting.md)
- [Contributing](docs/contributing.md) · [Experiment log](docs/experiment_log.md)

## Repository layout

```
countsampler/
├── novel/                  ★ RESEARCH CODE: everything we tune or change to get results
│   ├── shared.py           §2   sampler Δφ + geometry losses (one file on purpose)
│   ├── losses/             one file per method: spectral.py (A), density.py (B), peak.py (C)
│   ├── components/         counter.py, the density counter C_ψ (B)
│   └── methods/            loss weighting + stage schedules (Eq. 15, 17, 22)
├── generator/              frozen SD-Turbo / SDXL-Turbo wrapper, attention capture, Eq. (1)
├── data/                   prompts, synthetic shapes, counter datasets, generator harvesting
├── training/               sampler loop (§7), counter training (§4.1), checkpoints
├── evaluation/             hard counters, metrics, evaluation (§8), pre-checks / offline scan
└── utils/                  config, logging, misc
configs/                    YAML. Every spec knob is here, and open choices (§9) are marked OPEN
scripts/                    CLI entry points
docs/                       documentation
tests/                      CPU tests on tiny random models + docs drift checks
```

Every file in `countsampler/novel/` starts with a **★ NOVEL ★** banner listing the equations it implements and where it comes from (§10).

## Quick start (GPU machine)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -e ".[dev]"
```
```bash
python scripts/precheck.py --config configs/base.yaml --methods A C
```
```bash
python scripts/train_sampler.py --config configs/method_a.yaml
```
```bash
python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt
```

For the §8 benchmarks: `python scripts/prepare_benchmarks.py`, then add `--benchmark cococount` (or `d2d_small`, `d2d_multi`, `d2d_large`) to `evaluate.py`.
Method B needs a counter first (`python scripts/train_counter.py --config configs/counter.yaml`).
On 8 GB cards add `--config configs/gpu_8gb.yaml`. The full protocol is in [docs/workflow.md](docs/workflow.md).

## Tests

```bash
pytest -q
```

Runs on the CPU in about 20 seconds with no downloads. Covers the losses, gradient flow through the frozen U-Net, equivalence with
diffusers' SD-Turbo pipeline, end-to-end training steps for A/B/C, and checks that the docs stay in sync with the code.
