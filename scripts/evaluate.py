"""Evaluate a trained sampler, or the frozen baseline, with an independent hard counter (spec §8).

    python scripts/evaluate.py --ckpt runs/method_a/sampler_last.pt
    python scripts/evaluate.py --baseline --config configs/base.yaml --out-dir runs/eval_baseline
    python scripts/evaluate.py --ckpt runs/method_b/sampler_last.pt --split all --counter owlv2 --clip
Then compare: python scripts/compare.py runs/eval_baseline runs/method_a/eval runs/method_b/eval
"""
from pathlib import Path

from _common import base_parser, config_from_args

from countsampler.evaluation import build_hard_counter, run_evaluation
from countsampler.generator import FrozenTurboGenerator
from countsampler.training import load_sampler_checkpoint
from countsampler.utils.config import deep_merge, to_dict, wrap
from countsampler.utils.misc import resolve_device, seed_everything


def main() -> None:
    ap = base_parser(__doc__, config_required=False)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", help="sampler checkpoint (its stored config is used unless --config is given)")
    group.add_argument("--baseline", action="store_true", help="frozen baseline: z = ε")
    ap.add_argument("--split", default=None, choices=["heldout", "train", "all"])
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default eval.seeds)")
    ap.add_argument("--counter", default=None, help="hard counter: color | owlv2 (default eval.hard_counter)")
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
        cfg = wrap(ckpt["config"])
        if args.set:
            from countsampler.utils.config import parse_override, set_dotted

            raw = to_dict(cfg)
            for ov in args.set:
                set_dotted(raw, *parse_override(ov))
            cfg = wrap(raw)
    else:
        ap.error("--baseline needs --config")
    cfg = wrap(deep_merge({"eval": {}}, to_dict(cfg)))

    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, device)
    if sampler is not None:
        sampler = sampler.to(gen.device)
    ev = cfg.eval
    counter = build_hard_counter(args.counter or ev.get("hard_counter", "color"))
    n_seeds = args.seeds or int(ev.get("seeds", 16))
    out_dir = args.out_dir or (Path(args.ckpt).parent / "eval" if args.ckpt else "runs/eval_baseline")
    metrics = run_evaluation(cfg, gen, sampler, counter, out_dir, split=args.split or ev.get("split", "heldout"),
                             seeds=list(range(n_seeds)), counts=ev.get("counts"),
                             save_images=args.save_images or bool(ev.get("save_images", False)),
                             clip=args.clip or bool(ev.get("clip", False)))
    print(f"\nresults written to {out_dir}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
