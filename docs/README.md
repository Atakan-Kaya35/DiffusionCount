# Documentation

Documentation for the CS485 count-conditioned noise sampler. The specification is
[`count_sampler_formalization_v5.pdf`](../count_sampler_formalization_v5.pdf). These pages explain how the code implements it.

## Reading order

**For the results, start with [RESULTS.md](RESULTS.md)** — the presentation-ready package:
headline tables, figures, baselines, costs and limitations in one place.

**For the full evidence trail, read [Findings](findings.md)** — what is established, with the measurement
behind each claim, including what beats the baseline and why the attention-based methods failed.
The running narrative with the wrong turns and corrections is in
[the experiment log](experiment_log.md).

New to the project? Read these in order:

1. [Overview](overview.md): the idea, the pipeline, how gradients reach the sampler, and how the three methods differ.
2. [Setup](setup.md): installing on an RTX 50-series machine and checking that it works.
3. [Workflow](workflow.md): the experiment protocol from pre-checks to the comparison table.

## Methods

| Page | Spec | Code |
|---|---|---|
| [Shared components](methods/shared.md) | §2 | `countsampler/novel/shared.py` |
| [Method A: spectral count](methods/method_a_spectral.md) | §3 | `countsampler/novel/losses/spectral.py` |
| [Method B: density counter](methods/method_b_density.md) | §4 | `countsampler/novel/losses/density.py`, `countsampler/novel/components/counter.py` |
| [Method C: peak separation](methods/method_c_peak.md) | §5 | `countsampler/novel/losses/peak.py` |

## Reference

- [Spec → code map](spec_map.md): every equation and section, with the function that implements it.
- [Benchmarks](benchmarks.md): CoCoCount and the D2D-style sets, counting protocol, how the sampler handles them.
- [Configuration reference](configuration.md): every config key, its default, and what it does.
- [Outputs and logs](outputs.md): what each run writes, every logged metric, and checkpoint formats.
- [Architecture](architecture.md): how the code fits together, and how to add a method, counter or generator.
- [Spec decisions](spec_decisions.md): where the spec was open or ambiguous, what we chose, and what is not implemented yet.

## Team

- [Contributing](contributing.md): git workflow, tests, and how to keep these docs accurate.
- [Experiment log](experiment_log.md): record every run here.
- [Troubleshooting](troubleshooting.md): common errors and what to do about them.
