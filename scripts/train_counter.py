"""Train the density counter C_ψ on synthetic shapes (spec §4.1). Run once, before Method B.

    python scripts/train_counter.py --config configs/counter.yaml
    python scripts/train_counter.py --config configs/counter.yaml --set counter.input_space=image
"""
from _common import base_parser, config_from_args

from countsampler.training import train_counter


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    metrics = train_counter(config_from_args(args), out_dir=args.out_dir)
    print("final validation:", metrics)


if __name__ == "__main__":
    main()
