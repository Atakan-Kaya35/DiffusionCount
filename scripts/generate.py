"""Qualitative grid: for each seed, a baseline row (z = ε) and a sampler row (z_n), with counts as columns.

    python scripts/generate.py --ckpt runs/method_a/sampler_last.pt --color red --shape circle --out grid.png
"""
import argparse

import torch
from _common import ROOT  # noqa: F401  (puts the repo on sys.path)

from countsampler.data.prompts import PromptSampler
from countsampler.generator import FrozenTurboGenerator
from countsampler.training import load_sampler_checkpoint
from countsampler.utils.config import wrap
from countsampler.utils.misc import resolve_device, save_image_grid


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--color", default="red")
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--counts", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    sampler, ckpt = load_sampler_checkpoint(args.ckpt)
    cfg = wrap(ckpt["config"])
    gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
    sampler = sampler.to(gen.device)
    prompter = PromptSampler(cfg.data)
    eps, prompts, ns, is_base = [], [], [], []
    for s in args.seeds:
        e = torch.randn(1, *gen.latent_shape, generator=torch.Generator().manual_seed(s))
        for base in (True, False):
            for n in args.counts:
                eps.append(e)
                prompts.append(prompter.prompt(n, args.color, args.shape)[0])
                ns.append(n)
                is_base.append(base)
    eps = torch.cat(eps).to(gen.device)
    enc = gen.encode_prompts(prompts)
    z = sampler(eps, torch.tensor(ns, device=gen.device), enc.pooled.float())
    z = torch.where(torch.tensor(is_base, device=gen.device)[:, None, None, None], eps, z)
    imgs = gen.decode(gen.generate_latents(z, enc))
    save_image_grid(torch.nn.functional.interpolate(imgs, scale_factor=0.5, mode="bilinear", antialias=True),
                    args.out, nrow=len(args.counts))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
