"""Evaluation protocol (spec §8): count accuracy with an independent hard counter, per-seed count
histograms (can a seed's preferred count be moved?), latency, diversity and layout regularity.

The same code runs the frozen baseline (sampler=None, so z = ε) and any trained sampler, so the
comparison table (baseline / Method A / B / C) uses one shared counter.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.prompts import PromptSampler
from ..utils.misc import ensure_dir, save_image_grid, to_uint8_images
from .hard_counters import count_images
from .metrics import ClipScorer, count_metrics, nn_regularity, pairwise_diversity


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    if len(set(y)) < 2 or len(set(x)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


@torch.no_grad()
def run_evaluation(cfg, generator, sampler, counter, out_dir: str | Path, *, split: str = "heldout",
                   seeds: list[int] | None = None, counts: list[int] | None = None,
                   colors: list[str] | None = None, shapes: list[str] | None = None,
                   save_images: bool = False, clip: bool = False, timesteps=None,
                   counter_batch: int = 8, template_index: int = 0,
                   grid_scale: float = 0.5) -> dict:
    out_dir = ensure_dir(out_dir)
    gen = generator
    device = gen.device
    data = cfg.data
    seeds = list(seeds if seeds is not None else range(8))
    counts = list(counts or range(1, int(data.n_max) + 1))
    colors = list(colors or data.colors)
    if shapes is None:
        shapes = {"heldout": list(data.heldout_shapes), "train": list(data.train_shapes),
                  "all": list(data.train_shapes) + list(data.heldout_shapes)}[split]
    heldout_set = set(data.heldout_shapes)
    prompter = PromptSampler(data, split="train")
    clip_scorer = ClipScorer(device=device) if clip else None
    if sampler is not None:
        sampler.eval()

    # Training samples a template at random from data.templates; evaluation used templates[0]
    # unconditionally, so the sampler was scored on one phrasing out of the three it saw. That
    # phrasing is also the bare noun phrase, which this generator answers with edge-to-edge
    # tiling rather than countable objects.
    tmpl = list(data.templates)
    template = tmpl[int(template_index) % len(tmpl)]

    records, latencies, diversities = [], [], []
    for color in colors:
        for shape in shapes:
            eps_list, prompts, ns, sds = [], [], [], []
            for s in seeds:
                e = torch.randn(1, *gen.latent_shape, generator=torch.Generator().manual_seed(int(s)))
                for n in counts:
                    eps_list.append(e)
                    prompts.append(prompter.prompt(n, color, shape, template)[0])
                    ns.append(n)
                    sds.append(s)
            eps = torch.cat(eps_list).to(device)
            enc = gen.encode_prompts(prompts)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            z = eps if sampler is None else sampler(eps, torch.tensor(ns, device=device), enc.pooled.float())
            imgs = gen.decode(gen.generate_latents(z, enc, timesteps))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) / len(ns))
            u8 = to_uint8_images(imgs)
            clip_vals = clip_scorer(u8, prompts) if clip_scorer else [float("nan")] * len(ns)
            # Every image in this (colour, shape) block shares one detector query, so they can go
            # through the counter in batches. One forward per image dominated the run time before.
            results = []
            for s0 in range(0, len(ns), counter_batch):
                results += count_images(counter, list(u8[s0:s0 + counter_batch]), color, shape)
            for i in range(len(ns)):
                res = results[i]
                records.append({"color": color, "shape": shape, "heldout_shape": shape in heldout_set,
                                "seed": sds[i], "n": ns[i], "counted": res.count,
                                "regularity": nn_regularity(res.centers), "clip": float(clip_vals[i])})
            for n in counts:
                idx = [i for i in range(len(ns)) if ns[i] == n]
                diversities.append(pairwise_diversity(imgs[idx]))
            if save_images:
                # The counter above reads the FULL decode; the grid is what any later re-count
                # reads. At grid_scale 0.5 those are different images - OWLv2 rescales its input
                # to 960 px, so a 256 px cell is upscaled 3.75x and a 512 px cell 1.875x, and the
                # detector has no reason to count them alike. Re-counting a 0.5 grid and comparing
                # with a records.csv therefore mixes resolution into whatever else changed.
                # grid_scale 1.0 saves the images that were actually counted.
                out_imgs = imgs if grid_scale == 1.0 else torch.nn.functional.interpolate(
                    imgs, scale_factor=grid_scale, mode="bilinear", antialias=True)
                save_image_grid(out_imgs, out_dir / "grids" / f"{color}_{shape}.png", nrow=len(counts))

    pred = [r["counted"] for r in records]
    tgt = [r["n"] for r in records]
    metrics = {"overall": count_metrics(pred, tgt)}
    for flag, name in ((True, "heldout_shapes"), (False, "train_shapes")):
        sub = [r for r in records if r["heldout_shape"] == flag]
        if sub:
            metrics[name] = count_metrics([r["counted"] for r in sub], [r["n"] for r in sub])
    regs = [r["regularity"] for r in records if r["regularity"] is not None]
    clips = [r["clip"] for r in records if not np.isnan(r["clip"])]
    metrics["latency_ms_per_image"] = 1000 * float(np.mean(latencies))
    metrics["diversity"] = float(np.nanmean(diversities)) if diversities else float("nan")
    metrics["nn_regularity_cv"] = float(np.mean(regs)) if regs else float("nan")
    if clips:
        metrics["clip_score"] = float(np.mean(clips))

    # Per-seed count histograms: does the counted value follow the requested n for a fixed seed?
    per_seed = {}
    for r in records:
        per_seed.setdefault((r["seed"], r["color"], r["shape"]), []).append((r["n"], r["counted"]))
    rhos, distinct = [], []
    with open(out_dir / "per_seed_hist.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "color", "shape"] + [f"n={n}" for n in counts] + ["spearman", "distinct_counts"])
        for (s, c, sh), pairs in sorted(per_seed.items()):
            pairs.sort()
            got = [g for _, g in pairs]
            rho = _spearman([n for n, _ in pairs], got)
            rhos.append(rho)
            distinct.append(len(set(got)))
            w.writerow([s, c, sh] + got + [rho, len(set(got))])
    finite_rhos = [r for r in rhos if not np.isnan(r)]  # NaN when a seed's counts never vary
    metrics["seed_spearman_mean"] = float(np.mean(finite_rhos)) if finite_rhos else float("nan")
    metrics["seed_distinct_counts_mean"] = float(np.mean(distinct)) if distinct else float("nan")
    metrics["template"] = template

    with open(out_dir / "records.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
