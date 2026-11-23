# Benchmarks (spec §8)

Spec §8 asks for a comparison table of "frozen baseline, D2D, Method A, Method B, Method C, each on CoCoCount and the
D2D benchmarks with one shared counter, reporting accuracy, per-image latency and image quality".
The synthetic shape prompts ([workflow.md](workflow.md#4-evaluate)) show whether the method works where it was trained.
The benchmarks show whether it transfers to real object classes.

Code: `countsampler/data/benchmarks.py` (loaders and D2D-style builder), `countsampler/evaluation/benchmark_eval.py`
(evaluation), `scripts/prepare_benchmarks.py`, `scripts/evaluate.py --benchmark`.

## The benchmarks

| Name | Source | Prompts | Objects | Counts | Status here |
|---|---|---|---|---|---|
| `cococount` | Make It Count (Binyamin et al., CVPR 2025) | 200 ("A photo of {number} {objects} [on the grass / on the road / on the ground]") | 20 COCO classes | 2, 3, 4, 5, 7, 10 | **official file**, downloaded by `scripts/prepare_benchmarks.py` |
| `d2d_small` | D2D (Yoo et al., arXiv 2510.19278, §4.1) | 400 ("A realistic photo of a scene with [count] [object].") | 40 countable COCO classes | 1–10 | **reconstruction**, see below |
| `d2d_multi` | D2D | 400 ("… with [count] [object] and [count] [object].") | 2 different classes per prompt | each 1–9 | **reconstruction** |
| `d2d_large` | D2D | 400 | 40 classes | 11–20 | **reconstruction** |
| `jsonl` | your own file | any | any | any | — |

**CoCoCount** comes from `dataset/CoCoCount.json` in the [Make It Count repository](https://github.com/Litalby1/make-it-count).
That repository states no licence, so the file is downloaded to `data_cache/benchmarks/` (gitignored) rather than committed.
Each record carries its own seed. Some prompts appear several times with different seeds (161 distinct prompts in the release,
according to later work). `eval.benchmark.dedupe=true` keeps each prompt once.

**D2D-style sets.** D2D's benchmark files are not public yet (their repository was "under construction" when we checked).
`build_d2d_style` rebuilds them from the paper's description: the templates, sizes and count ranges match, but the
paper does not list its 40 classes. We use the 20 CoCoCount classes plus 20 other countable COCO classes
(`D2D_STYLE_CLASSES`). D2D-Large is described only as "similarly constructed with counts 11–20". We read that as D2D-Small with counts 11–20.
D2D-Multi's lower count bound is not given, so we use 1–9. **Report these as "D2D-style", not as official D2D numbers.**
The generated files are committed in `benchmarks/`, so every teammate evaluates exactly the same prompts.
`tests/test_benchmarks.py` checks that they still match the builder.

**Custom prompt files** (`--benchmark jsonl --benchmark-path file.jsonl`), one JSON object per line:

```json
{"prompt": "A photo of three dogs on a sofa", "object": "dog", "count": 3}
{"prompt": "two cats and one dog", "objects": [{"name": "cat", "count": 2}, {"name": "dog", "count": 1}], "seed": 7}
```

`name` is the class the counter looks for. For the YOLO counter it must be a COCO class name.

## Running

```bash
python scripts/prepare_benchmarks.py
```
```bash
python scripts/evaluate.py --baseline --config configs/base.yaml --benchmark cococount
```
```bash
python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt --benchmark cococount
```
```bash
python scripts/compare.py runs/eval_baseline_cococount runs/method_a/eval_cococount
```

Outputs go to `runs/eval_baseline_<benchmark>/` or `runs/<run>/eval_<benchmark>/`. See [outputs.md](outputs.md#benchmark-evaluation).

## Counting protocol

| Counter | `--counter` | Used by | Notes |
|---|---|---|---|
| OWLv2 | `owlv2` (default) | D2D's training critic | open vocabulary, query "a photo of a {object}", score threshold 0.3, NMS 0.5 |
| YOLOv9 | `yolo` | Make It Count's CoCoCount evaluation | `pip install -e ".[yolo]"` (ultralytics, AGPL-3.0). Counts boxes whose COCO class name equals the object |
| CountGD | external | D2D's evaluation | not pip-installable. Run with `--save-images` and count the images listed in `manifest.jsonl` with CountGD |

Use **one counter for every row** of a comparison table, as §8 requires.

**Scoring.** Each requested object is counted separately. An image is correct only if **every** object in its prompt has the requested
count. For single-object prompts this is plain exact accuracy. D2D does not say how it scores multi-object prompts, and strict
scoring is the natural reading of "proportion of correctly-generated images". Accuracy is computed per seed and reported as the mean ± std over
seeds (`eval.benchmark.seeds`, default 4, as in D2D). Per-object accuracy and MAE are reported too.

## How the sampler handles these prompts

The sampler was trained on synthetic "n colour shape" prompts. On benchmarks it gets the benchmark prompt's c̄ and the requested n,
so the benchmark measures **transfer** to unseen classes and phrasings. Two cases fall outside what the spec defines:

- **Multi-object prompts** (D2D-Multi): Eq. 2 takes a single count. `multi_object_count=sum` conditions on the total number of
  objects. `skip` leaves these prompts out (for the baseline too, so the rows stay comparable). Results are reported separately as `multi_object`.
- **Counts above `data.n_max`** (D2D-Large, 11–20): the sampler never saw them, and its sinusoidal log n encoding has to extrapolate.
  These are reported separately as `out_of_train_range`. To target them, train with a larger `data.n_max`.

## Not implemented

- **D2D itself** as a baseline: run the authors' code once it is released and count its images with the same counter.
- **The official D2D benchmark files**: when they are published, convert them to the JSONL format above (or add a loader next to `load_cococount`).
