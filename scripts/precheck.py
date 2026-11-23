"""Method-specific pre-checks on the frozen model (spec §8). Also the offline scan for ℓ, t, γ, τ, T (§9).

    python scripts/precheck.py --config configs/base.yaml --methods A C
    python scripts/precheck.py --config configs/base.yaml --methods B --counter-ckpt runs/counter/counter_best.pt
    python scripts/precheck.py --config configs/base.yaml --methods A --set precheck.sources=[synthetic]
Writes precheck_{A,B,C}_{rows,summary}.csv to --out-dir and prints the best configurations.
"""
from _common import base_parser, config_from_args

from countsampler.evaluation import build_hard_counter, run_prechecks
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.components import load_counter
from countsampler.utils.misc import resolve_device, seed_everything


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="+", default=["A", "C"], choices=["A", "B", "C"])
    ap.add_argument("--counter-ckpt", default=None, help="needed for method B")
    ap.add_argument("--out-dir", default="runs/precheck")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, device)
    hard = build_hard_counter(cfg.precheck.get("hard_counter", "color"))
    counter, meta = (load_counter(args.counter_ckpt, device) if args.counter_ckpt else (None, None))
    run_prechecks(cfg, gen, [m.upper() for m in args.methods], args.out_dir, hard_counter=hard,
                  counter=counter, counter_meta=meta)


if __name__ == "__main__":
    main()
