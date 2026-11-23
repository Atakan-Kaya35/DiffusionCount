# Contributing

## Git workflow

- `main` should always pass `pytest -q`.
- Work on a branch named `<your-name>/<topic>` (e.g. `ayse/method-c-init`) and merge through a pull request once the GitHub repo exists.
- Commit small and often, with messages that say *why*: `Lower tau_a default: precheck shows ā rarely exceeds 0.3 at t=999`.
- Never commit `runs/`, `data_cache/`, checkpoints (`*.pt`) or the lecture slides in `prev_taught/`. `.gitignore` covers them. Share checkpoints out of band.

## Before you push

```bash
pytest -q
```

The suite runs on the CPU in about 10 seconds with no downloads. It includes `tests/test_docs.py`, which fails if:
- a config key in `configs/*.yaml` is missing from [configuration.md](configuration.md);
- a relative link in `docs/` or `README.md` points to a file that does not exist;
- a backticked repository path in the docs (e.g. `countsampler/novel/shared.py`) does not exist.

## Where code goes

- **Research code** (losses, sampler, counter, objectives) goes in `countsampler/novel/`. Start each new file with the ★ NOVEL ★ banner:
  which spec section and equations it implements, and where the idea comes from. Put equation numbers in docstrings.
- **Plumbing** (generator, data, training loop, evaluation) goes outside `novel/`.
- **New experiments** should be config changes when possible. If a run matters for the report, save its overrides as a
  versioned overlay, e.g. `configs/experiments/a_glin.yaml`, instead of a long `--set` line nobody can reproduce.

## Keeping docs current

| If you change… | Update… |
|---|---|
| a config key or default | the YAML comment and [configuration.md](configuration.md) (the test checks presence, not the default value) |
| a loss or method behaviour | the method page in `docs/methods/` and [spec_map.md](spec_map.md) |
| an interpretation of the spec | [spec_decisions.md](spec_decisions.md) |
| a log key or output file | [outputs.md](outputs.md) |

## Recording results

Every run that is not a smoke test gets a row in [experiment_log.md](experiment_log.md), including failed runs. A method
that does not work on the frozen model (e.g. pre-check C finds no separate peaks) is a result for the report.
