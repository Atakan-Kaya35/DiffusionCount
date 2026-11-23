"""Harvest frozen-generator x̂0 latents labelled by a hard counter, as extra counter training data (§4.1).

    python scripts/harvest_generator_samples.py --config configs/counter.yaml --out data_cache/counter/harvest.pt
    python scripts/train_counter.py --config configs/counter.yaml --set "counter.dataset.harvested=[data_cache/counter/harvest.pt]"
"""
from _common import base_parser, config_from_args

from countsampler.data.harvest import harvest_generator_samples
from countsampler.evaluation import build_hard_counter
from countsampler.generator import FrozenTurboGenerator
from countsampler.utils.misc import resolve_device


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--counter", default="color")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    cfg = config_from_args(args)
    gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
    stats = harvest_generator_samples(cfg, gen, build_hard_counter(args.counter), args.out,
                                      samples_per_prompt=args.samples_per_prompt, batch_size=args.batch_size,
                                      seed=int(cfg.experiment.seed))
    print(stats)


if __name__ == "__main__":
    main()
