"""Train the density counter C_ψ once, before Method B sampler training (spec §4.1).

Loss = MSE(s·D, s·target) + λ_L1 · |Σ D_k − n_k|. The MSE is on the density scaled by s,
because the Gaussian targets are tiny. Soft counts are therefore Σ D_k = n_k, not s·n_k.
Validation: count MAE and exact-count accuracy on held-out images, overall, on present
categories, and on held-out (category, count) combinations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..data.counter_data import (
    CounterSceneSampler,
    ImageCountingDataset,
    LatentCountingDataset,
    augment_batch,
    build_latent_cache,
    counter_categories,
    heldout_combos,
)
from ..data.shapes import ShapeRenderer
from ..generator.turbo import load_vae, vae_encode
from ..novel.components.counter import DensityCounter, save_counter
from ..utils.config import to_dict
from ..utils.misc import ensure_dir, resolve_device, seed_everything
from ..utils.run_logger import RunLogger


def _cache_key(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


@torch.no_grad()
def evaluate_counter(model, loader, device, categories, heldout: set) -> dict:
    model.eval()
    preds, trues = [], []
    for x, _dens, counts in loader:
        preds.append(model.count(x.to(device).float()).cpu())
        trues.append(counts)
    p, t = torch.cat(preds), torch.cat(trues)
    err = (p - t).abs()
    exact = (p.round() == t).float()
    present = t > 0
    ho = torch.zeros_like(present)
    for k, cat in enumerate(categories):
        for n in range(1, int(t.max().item()) + 1):
            if (cat, n) in heldout:
                ho[:, k] |= t[:, k] == n
    out = {
        "mae_all": err.mean().item(),
        "exact_all": exact.mean().item(),
        "mae_present": err[present].mean().item() if present.any() else float("nan"),
        "exact_present": exact[present].mean().item() if present.any() else float("nan"),
        "stray_mass": p[~present].mean().item() if (~present).any() else 0.0,
    }
    if ho.any():
        out["mae_heldout_combo"] = err[ho].mean().item()
        out["exact_heldout_combo"] = exact[ho].mean().item()
    model.train()
    return out


def train_counter(cfg, out_dir: str | Path | None = None) -> dict:
    exp, ccfg, ds = cfg.experiment, cfg.counter, cfg.counter.dataset
    seed = int(exp.seed)
    seed_everything(seed)
    device = resolve_device(exp.get("device", "auto"))
    out_dir = ensure_dir(out_dir or Path(exp.output_dir) / exp.name)
    logger = RunLogger(out_dir, use_wandb=bool(exp.get("wandb", False)), run_name=exp.name, config=to_dict(cfg))

    colors = list(ccfg.get("colors") or cfg.data.colors)
    shapes = list(ccfg.get("shapes") or (list(cfg.data.train_shapes) + list(cfg.data.heldout_shapes)))
    categories = counter_categories(colors, shapes)
    n_max = int(cfg.data.n_max)
    heldout = heldout_combos(categories, n_max, float(ds.heldout_combo_fraction), seed)
    sampler_kw = dict(n_max=n_max, max_categories=int(ds.max_categories_per_image), max_total=int(ds.max_total_objects),
                      empty_prob=float(ds.empty_prob), touching_prob=float(ds.touching_prob),
                      identical_prob=float(ds.identical_prob), heldout=heldout)
    train_sampler = CounterSceneSampler(categories, split="train", **sampler_kw)
    val_sampler = CounterSceneSampler(categories, split="val", **sampler_kw)
    rr = tuple(ds.get("radius_range", (0.045, 0.09)))
    input_space = ccfg.get("input_space", "latent")
    sigma_frac = float(ccfg.density_sigma_frac)

    if input_space == "latent":
        res = int(cfg.generator.get("resolution", 512))
        renderer = ShapeRenderer(size=res, radius_range=rr, background=ds.get("background", "white"))
        vae = load_vae(cfg.generator, device)
        key_base = {"model": cfg.generator.model_id, "res": res, "cats": categories, "seed": seed,
                    "rr": list(rr), "ds": to_dict(ds), "aug": to_dict(ccfg.get("image_aug") or {})}
        cache_dir = ensure_dir(ds.cache_dir)
        paths = {}
        for split, smp, num, sd in (("train", train_sampler, int(ds.train_size), seed),
                                    ("val", val_sampler, int(ds.val_size), seed + 7919)):
            p = cache_dir / f"{split}_{_cache_key({**key_base, 'split': split})}.pt"
            if not p.exists():
                build_latent_cache(p, smp, renderer, categories, num, sd, lambda x: vae_encode(vae, x),
                                   image_aug=(ccfg.get("image_aug") if split == "train" else None))
            paths[split] = p
        del vae
        sigma = sigma_frac * (res // 8)
        train_sets = [LatentCountingDataset(paths["train"], categories, sigma, flip=True)]
        for extra in ds.get("harvested", []) or []:
            train_sets.append(LatentCountingDataset(extra, categories, sigma, flip=True))
        train_ds = ConcatDataset(train_sets)
        val_ds = LatentCountingDataset(paths["val"], categories, sigma, flip=False)
        in_channels = int(ccfg.get("latent_channels", 4))
        image_size = None
    elif input_space == "image":
        image_size = int(ccfg.image_size)
        renderer = ShapeRenderer(size=image_size, radius_range=rr, background=ds.get("background", "white"))
        sigma = sigma_frac * image_size
        train_ds = ImageCountingDataset(train_sampler, renderer, categories, int(ds.train_size), seed, sigma)
        val_ds = ImageCountingDataset(val_sampler, renderer, categories, int(ds.val_size), seed + 7919, sigma)
        in_channels = 3
    else:
        raise ValueError("counter.input_space must be 'latent' or 'image'")

    tc = ccfg.train
    workers = int(tc.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=int(tc.batch_size), shuffle=True, num_workers=workers,
                              drop_last=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=int(tc.batch_size), shuffle=False, num_workers=workers)

    arch = to_dict(ccfg.arch)
    model = DensityCounter(in_channels, len(categories), base_channels=arch["base_channels"],
                           channel_mult=tuple(arch["channel_mult"]), out_activation=arch["out_activation"],
                           head_bias_init=arch["head_bias_init"]).to(device)
    epochs = int(tc.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc.lr), weight_decay=float(tc.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(train_loader)))
    s = float(tc.mse_scale)
    l1_w = float(tc.l1_weight)
    aug = to_dict(ccfg.get("online_aug") or {})
    meta = {"in_channels": in_channels, "categories": categories, "input_space": input_space,
            "image_size": image_size, "arch": arch, "density_sigma": sigma,
            "heldout_combos": sorted([list(c) for c in heldout]), "model_id": cfg.generator.model_id}

    best, step, metrics = float("inf"), 0, {}
    for epoch in range(epochs):
        model.train()
        for x, dens, counts in train_loader:
            x, dens, counts = x.to(device), dens.to(device), counts.to(device)
            if aug:
                x = augment_batch(x, aug)
            pred = model(x)
            mse = ((s * pred - s * dens) ** 2).mean()
            l1 = (pred.sum(dim=(-2, -1)) - counts).abs().mean()
            loss = mse + l1_w * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % int(tc.get("log_every", 50)) == 0:
                logger.log(step, {"epoch": epoch, "loss": loss.item(), "mse": mse.item(), "l1": l1.item(),
                                  "lr": opt.param_groups[0]["lr"]})
        metrics = evaluate_counter(model, val_loader, device, categories, heldout)
        logger.log(step, {"epoch": epoch, **{f"val/{k}": v for k, v in metrics.items()}})
        save_counter(out_dir / "counter_last.pt", model, {**meta, "metrics": metrics, "epoch": epoch})
        if metrics["mae_present"] < best:
            best = metrics["mae_present"]
            save_counter(out_dir / "counter_best.pt", model, {**meta, "metrics": metrics, "epoch": epoch})
    logger.close()
    return metrics
