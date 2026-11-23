"""List the U-Net's attention layers (name, self/cross, token grid), for filling method.attention.*.

    python scripts/list_layers.py --config configs/base.yaml
    python scripts/list_layers.py --config configs/base.yaml --config configs/sdxl_turbo.yaml
"""
from _common import base_parser, config_from_args

from countsampler.generator import FrozenTurboGenerator
from countsampler.utils.misc import resolve_device


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
    info = gen.attention_layers()
    print(f"{'kind':<6} {'grid':<8} name")
    for name, li in info.items():
        print(f"{li.kind:<6} {li.grid[0]}x{li.grid[1]:<5} {name}")


if __name__ == "__main__":
    main()
