"""Benchmark evaluation (spec §8): CoCoCount, D2D-style sets or any prompt file, scored with a detector.

Each (prompt, seed) produces one image from z = ε (baseline) or z = sampler(ε, n, c̄) (trained).
Scoring:
  * per object: the detector's count for that class vs the requested count;
  * per image: correct only if EVERY object in the prompt has the right count (strict, the natural
    reading of "proportion of correctly-generated images" for multi-object prompts);
  * accuracy per seed, then mean ± std over seeds (D2D reports the average over four seeds).

The sampler takes one count, so multi-object prompts (D2D-Multi) need a policy
(`multi_object_count`): "sum" conditions it on the total number of objects, "skip" drops these prompts.
Counts above data.n_max are outside the sampler's training range and are reported separately.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.benchmarks import BenchmarkItem
from ..utils.misc import ensure_dir, to_uint8_images
from .metrics import ClipScorer, count_metrics


def _image_acc(images: list[dict]) -> dict:
    return {"image_exact_acc": float(np.mean([im["correct"] for im in images])) if images else float("nan"),
            "num_images": len(images)}


@torch.no_grad()
def run_benchmark_evaluation(cfg, generator, sampler, counter, items: list[BenchmarkItem], out_dir: str | Path, *,
                             benchmark_name: str = "benchmark", seeds: int | str = 4,
                             multi_object_count: str = "sum", batch_size: int = 8, save_images: bool = False,
                             clip: bool = False, timesteps=None) -> dict:
    if multi_object_count not in ("sum", "skip"):
        raise ValueError("multi_object_count must be 'sum' or 'skip'")
    out_dir = ensure_dir(out_dir)
    gen = generator
    device = gen.device
    n_max = int(cfg.data.n_max)
    if multi_object_count == "skip":
        items = [it for it in items if len(it.objects) == 1]
    jobs = []  # (item index, seed label, seed value)
    for i, it in enumerate(items):
        if seeds == "record":
            if it.seed is None:
                raise ValueError(f"item {i} has no seed; use an integer number of seeds")
            jobs.append((i, "record", int(it.seed)))
        else:
            jobs.extend((i, s, s) for s in range(int(seeds)))
    if sampler is not None:
        sampler.eval()
    clip_scorer = ClipScorer(device=device) if clip else None
    img_dir = ensure_dir(out_dir / "images") if save_images else None

    records, images, latencies = [], [], []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        chunk_items = [items[i] for i, _, _ in chunk]
        prompts = [it.prompt for it in chunk_items]
        eps = torch.cat([torch.randn(1, *gen.latent_shape, generator=torch.Generator().manual_seed(sv))
                         for _, _, sv in chunk]).to(device)
        n_cond = torch.tensor([it.total_count for it in chunk_items], device=device)
        enc = gen.encode_prompts(prompts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        z = eps if sampler is None else sampler(eps, n_cond, enc.pooled.float())
        imgs = gen.decode(gen.generate_latents(z, enc, timesteps))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) / len(chunk))
        u8 = to_uint8_images(imgs)
        clip_vals = clip_scorer(u8, prompts) if clip_scorer else [float("nan")] * len(chunk)

        for j, ((i, s_label, s_val), it) in enumerate(zip(chunk, chunk_items)):
            counted = [counter.count(u8[j], None, o.name).count for o in it.objects]
            correct = all(c == o.count for c, o in zip(counted, it.objects))
            fname = None
            if img_dir is not None:
                from PIL import Image

                fname = f"{i:04d}_s{s_val}.png"
                Image.fromarray(u8[j]).save(img_dir / fname)
            images.append({"item": i, "seed": s_label, "correct": correct, "n_cond": it.total_count,
                           "in_range": max(o.count for o in it.objects) <= n_max,
                           "multi": len(it.objects) > 1, "clip": float(clip_vals[j]), "file": fname,
                           "prompt": it.prompt, "objects": it.objects})
            for o, c in zip(it.objects, counted):
                records.append({"item": i, "prompt": it.prompt, "seed": s_val, "object": o.name, "target": o.count,
                                "counted": c, "image_correct": correct, "n_cond": it.total_count,
                                "in_train_range": o.count <= n_max, "num_objects": len(it.objects),
                                "image_file": fname or ""})

    metrics: dict = {"benchmark": benchmark_name, "num_prompts": len(items), "seeds": seeds,
                     "multi_object_count": multi_object_count, **_image_acc(images)}
    per_seed = defaultdict(list)
    for im in images:
        per_seed[str(im["seed"])].append(im["correct"])
    accs = {s: float(np.mean(v)) for s, v in per_seed.items()}
    metrics["image_exact_acc_per_seed"] = accs
    metrics["image_exact_acc_mean_over_seeds"] = float(np.mean(list(accs.values()))) if accs else float("nan")
    metrics["image_exact_acc_std_over_seeds"] = float(np.std(list(accs.values()))) if accs else float("nan")
    metrics["object"] = count_metrics([r["counted"] for r in records], [r["target"] for r in records])
    metrics["in_train_range"] = _image_acc([im for im in images if im["in_range"]])
    metrics["out_of_train_range"] = _image_acc([im for im in images if not im["in_range"]])
    if any(im["multi"] for im in images):
        metrics["single_object"] = _image_acc([im for im in images if not im["multi"]])
        metrics["multi_object"] = _image_acc([im for im in images if im["multi"]])
    metrics["latency_ms_per_image"] = 1000 * float(np.mean(latencies)) if latencies else float("nan")
    clips = [im["clip"] for im in images if not np.isnan(im["clip"])]
    if clips:
        metrics["clip_score"] = float(np.mean(clips))

    with open(out_dir / "records.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()) if records else ["item"])
        w.writeheader()
        w.writerows(records)
    if img_dir is not None:  # manifest for counting the saved images with an external counter (e.g. CountGD)
        with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
            for im in images:
                fh.write(json.dumps({"file": f"images/{im['file']}", "prompt": im["prompt"], "seed": im["seed"],
                                     "objects": [{"name": o.name, "count": o.count} for o in im["objects"]]}) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
