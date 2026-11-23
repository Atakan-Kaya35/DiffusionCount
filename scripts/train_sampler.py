"""Train the count-conditioned sampler Δφ with Method A, B or C (spec §7).

    python scripts/train_sampler.py --config configs/method_a.yaml
    python scripts/train_sampler.py --config configs/method_b.yaml --set train.steps=8000
    python scripts/train_sampler.py --config configs/method_c.yaml --config configs/gpu_8gb.yaml
    # ablation (§7): number removed from the prompt
    python scripts/train_sampler.py --config configs/method_a.yaml --set data.include_count=false experiment.name=a_nonum
"""
from _common import base_parser, config_from_args

from countsampler.training import SamplerTrainer


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--out-dir", default=None, help="default: <experiment.output_dir>/<experiment.name>")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    args = ap.parse_args()
    cfg = config_from_args(args)
    trainer = SamplerTrainer(cfg, out_dir=args.out_dir)
    trainer.fit(args.steps)


if __name__ == "__main__":
    main()
