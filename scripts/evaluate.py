"""Evaluate a trained sampler, or the frozen baseline, with an independent hard counter (spec §8).

Synthetic shape prompts (default):
    python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt
    python scripts/evaluate.py --baseline --config configs/base.yaml --out-dir runs/eval_baseline
    python scripts/evaluate.py --ckpt runs/method_b/sampler_last.pt --split all --counter owlv2 --clip
Benchmarks (CoCoCount, D2D-style sets, or a custom JSONL; detector counter, OWLv2 by default):
    python scripts/prepare_benchmarks.py           # once: downloads CoCoCount, writes D2D-style files
    python scripts/evaluate.py --baseline --config configs/base.yaml --benchmark cococount
    python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt --benchmark d2d_small --counter yolo
    python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt --benchmark jsonl --benchmark-path my.jsonl
Then compare: python scripts/compare.py runs/eval_baseline runs/method_a/eval ...
"""
from pathlib import Path

from _common import base_parser, config_from_args

from countsampler.data.benchmarks import BENCHMARKS, load_benchmark
from countsampler.evaluation import build_hard_counter, run_benchmark_evaluation, run_evaluation
from countsampler.generator import FrozenTurboGenerator
from countsampler.training import load_sampler_checkpoint
from countsampler.utils.config import deep_merge, parse_override, set_dotted, to_dict, wrap
from countsampler.utils.misc import resolve_device, seed_everything


def main() -> None:
    ap = base_parser(__doc__, config_required=False)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", help="sampler checkpoint (its stored config is used unless --config is given)")
    group.add_argument("--baseline", action="store_true", help="frozen baseline: z = ε")
    ap.add_argument("--split", default=None, choices=["heldout", "train", "all"], help="synthetic mode only")
    ap.add_argument("--seeds", default=None,
                    help="number of seeds (default eval.seeds / eval.benchmark.seeds); 'record' = CoCoCount's own seeds")
    ap.add_argument("--counter", default=None, help="hard counter: color | owlv2 | yolo")
    ap.add_argument("--benchmark", default=None, choices=BENCHMARKS, help="evaluate on a benchmark instead")
    ap.add_argument("--benchmark-path", default=None, help="benchmark file (default per benchmark)")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sampler, ckpt = None, None
    if args.ckpt:
        sampler, ckpt = load_sampler_checkpoint(args.ckpt, "cpu")
    if args.config:
        cfg = config_from_args(args)
    elif ckpt is not None:
        raw = ckpt["config"]
        for ov in args.set:
            set_dotted(raw, *parse_override(ov))
        cfg = wrap(raw)
    else:
        ap.error("--baseline needs --config")
    cfg = wrap(deep_merge({"eval": {"benchmark": {}}}, to_dict(cfg)))  # older checkpoints lack these keys

    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, device)
    if sampler is not None:
        sampler = sampler.to(gen.device)
    ev, bcfg = cfg.eval, cfg.eval.benchmark
    save_images = args.save_images or bool(ev.get("save_images", False))
    clip = args.clip or bool(ev.get("clip", False))
    run_root = Path(args.ckpt).parent if args.ckpt else Path("runs")
    bench = args.benchmark or bcfg.get("name")

    if bench:
        items = load_benchmark(bench, args.benchmark_path or bcfg.get("path"), dedupe=bool(bcfg.get("dedupe", False)))
        seeds = args.seeds or bcfg.get("seeds", 4)
        seeds = seeds if seeds == "record" else int(seeds)
        counter = build_hard_counter(args.counter or bcfg.get("counter", "owlv2"), device=gen.device,
                                     **to_dict(bcfg.get("counter_kwargs") or {}))
        out_dir = args.out_dir or (run_root / (f"eval_{bench}" if args.ckpt else f"eval_baseline_{bench}"))
        metrics = run_benchmark_evaluation(
            cfg, gen, sampler, counter, items, out_dir, benchmark_name=bench, seeds=seeds,
            multi_object_count=bcfg.get("multi_object_count", "sum"), batch_size=int(bcfg.get("batch_size", 8)),
            save_images=save_images or bool(bcfg.get("save_images", False)), clip=clip)
    else:
        counter = build_hard_counter(args.counter or ev.get("hard_counter", "color"),
                                     device=gen.device, **to_dict(ev.get("counter_kwargs") or {}))
        out_dir = args.out_dir or (run_root / "eval" if args.ckpt else "runs/eval_baseline")
        n_seeds = int(args.seeds or ev.get("seeds", 16))
        metrics = run_evaluation(cfg, gen, sampler, counter, out_dir, split=args.split or ev.get("split", "heldout"),
                                 seeds=list(range(n_seeds)), counts=ev.get("counts"), save_images=save_images,
                                 clip=clip, template_index=int(ev.get("template_index", 0)))
    print(f"\nresults written to {out_dir}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
