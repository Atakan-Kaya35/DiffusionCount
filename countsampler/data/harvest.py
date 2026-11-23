"""Harvest x̂0 samples from the frozen generator, labelled by a hard counter, as extra counter
training data (spec §4.1: "if available, images x̂0 from the frozen generator with true counts as
targets, because x̂0 is blurrier than real images").

The output uses the latent-cache format, so it can be listed under counter.dataset.harvested.
Only the prompted category is labelled. Stray objects of other colours stay unlabelled, which is
a small amount of label noise.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from tqdm import tqdm

from .prompts import build_prompt


@torch.no_grad()
def harvest_generator_samples(cfg, gen, hard_counter, out_path: str | Path, samples_per_prompt: int = 2,
                              batch_size: int = 8, seed: int = 0, min_count: int = 1) -> dict:
    data = cfg.data
    colors = list(data.colors)
    shapes = list(data.train_shapes) + list(data.heldout_shapes)
    jobs = [(c, s, n, k) for c in colors for s in shapes for n in range(1, int(data.n_max) + 1)
            for k in range(samples_per_prompt)]
    random.Random(seed).shuffle(jobs)
    latents, objects = [], []
    kept = 0
    tpl, fmt = data.templates[0], data.get("count_format", "word")
    for i in tqdm(range(0, len(jobs), batch_size), desc="harvest"):
        chunk = jobs[i:i + batch_size]
        prompts = [build_prompt(tpl, n, c, s, fmt, True)[0] for c, s, n, _ in chunk]
        enc = gen.encode_prompts(prompts)
        z = torch.randn(len(chunk), *gen.latent_shape, generator=torch.Generator().manual_seed(seed + i)).to(gen.device)
        x0 = gen.generate_latents(z, enc)
        imgs = gen.decode(x0)
        for j, (c, s, _n, _) in enumerate(chunk):
            res = hard_counter.count(imgs[j], c, s)
            if res.count < min_count or res.count > int(data.n_max):
                continue
            latents.append(x0[j].half().cpu())
            objects.append([(f"{c} {s}", float(x), float(y)) for x, y in res.centers])
            kept += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.stack(latents) if latents else torch.empty(0),
                "objects": objects, "categories": [f"{c} {s}" for c in colors for s in shapes]}, out_path)
    return {"generated": len(jobs), "kept": kept}
