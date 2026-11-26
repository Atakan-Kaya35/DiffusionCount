"""Train the count sampler by policy gradient, with the detector itself as the reward.

Why this exists. Methods A-D all optimise a DIFFERENTIABLE proxy for the count, and all three
proxies that were measured carry almost none of the quantity they stand in for, measured at the
level a per-image loss actually sees:

    Method C   peak separation      degenerate in practice: 0.19 objects per image, 82% empty
    Method D   chi on attention     +0.154 within-n against the detector
    Method B   density counter      +0.186 pooled, -0.007 within-n on the total

That is not a statement about the sampler. probe_perturbation.py --mode leverage showed a
displacement of 0.079, inside the measured safe radius, changes the count by a median of one
object - exactly the correction the task needs. The obstacle is the measurement, not the
mechanism, so this removes the requirement that the measurement be differentiable.

A score-function estimator needs no gradient through the generator or the counter. The policy is
the sampler plus exploration noise,

    z = eps + w * Delta_phi(eps, n, c_bar) + sigma * xi,      xi ~ N(0, I)

so log pi(z | eps, n) = -||z - eps - w*Delta||^2 / (2 sigma^2) + const, and

    grad_phi E[R] = E[ (R - b) * (w / sigma^2) * <z - eps - w*Delta, grad_phi Delta> ]

implemented as the surrogate loss -(A * (w/sigma^2) * sum(xi_detached * Delta)). The reward is
OWLv2 on the decoded image, which is the evaluation metric itself - at 185 ms per image batched
this costs about 1.5 s per step, so the proxy can simply be dropped rather than improved.

The count stays OUT of the prompt (data.include_count: false), so Delta_phi is the only carrier
of n and the experiment is falsifiable: any skill above the constant predictor is the sampler's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import base_parser, config_from_args

from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import build_sampler, normalized_distance
from countsampler.training.checkpoint import save_sampler_checkpoint
from countsampler.utils.misc import (
    ensure_dir,
    resolve_device,
    save_image_grid,
    seed_everything,
    to_uint8_images,
)


def main() -> None:
    ap = base_parser(__doc__, config_required=False)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--sigma", type=float, default=0.02,
                    help="exploration sd in normalised distance; d(z, eps+wD) equals this")
    ap.add_argument("--eps-per-batch", type=int, default=2)
    ap.add_argument("--counts-per-eps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--counter", default="owlv2")
    ap.add_argument("--sd-floor", type=float, default=25.0,
                    help="image sd below which the decode is treated as collapsed (measured)")
    ap.add_argument("--collapse-penalty", type=float, default=5.0)
    ap.add_argument("--baseline-ema", type=float, default=0.9)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--preview-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = config_from_args(args)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(cfg.experiment.get("device", "auto"))
    gen = FrozenTurboGenerator.from_config(cfg.generator, device)
    gen.freeze()
    sampler = build_sampler(cfg.sampler, gen.latent_channels, gen.pooled_dim).to(device)
    opt = torch.optim.AdamW(sampler.parameters(), lr=args.lr,
                            weight_decay=float(cfg.train.get("weight_decay", 0.0)))
    counter = build_hard_counter(args.counter, device=gen.device, threshold=0.1, query_color=False)
    prompts = PromptSampler(cfg.data)
    out_dir = ensure_dir(args.out_dir or Path(cfg.experiment.output_dir) / cfg.experiment.name)
    log = open(out_dir / "log.jsonl", "a", encoding="utf-8")

    c, h, w_ = gen.latent_shape
    n_max = int(cfg.data.n_max)
    wmix = float(cfg.sampler.get("w", 1.0))
    sigma = float(args.sigma)
    noise_gen = torch.Generator().manual_seed(int(cfg.experiment.seed) + 1)
    colors, shapes = list(cfg.data.colors), list(cfg.data.heldout_shapes) + list(cfg.data.train_shapes)
    baseline = {k: None for k in range(1, n_max + 1)}   # per-count EMA, the variance reducer
    t0 = time.time()

    for step in range(1, args.steps + 1):
        eps_list, counts, prom, words, cols, shps = [], [], [], [], [], []
        for _ in range(args.eps_per_batch):
            e = torch.randn(1, c, h, w_, generator=noise_gen)
            col = colors[int(torch.randint(len(colors), (1,), generator=noise_gen))]
            shp = shapes[int(torch.randint(len(shapes), (1,), generator=noise_gen))]
            ns = torch.randperm(n_max, generator=noise_gen)[:args.counts_per_eps] + 1
            for n in ns.tolist():
                eps_list.append(e)
                counts.append(n)
                p, word = prompts.prompt(n, col, shp)
                prom.append(p); words.append(word); cols.append(col); shps.append(shp)
        eps = torch.cat(eps_list).to(device)
        n_t = torch.tensor(counts, device=device)
        enc = gen.encode_prompts(prom)

        # ---- policy: deterministic mean plus exploration noise.
        # Sampled under no_grad so the sampler's autograd graph is NOT held alive across the
        # rollout - keeping it resident pushes an 8 GB card into shared memory and the step from
        # 3 s to 8.5 s. The graph is rebuilt for the update once the reward is known, at the cost
        # of one extra sampler forward (0.05 s).
        with torch.no_grad():
            delta_s = sampler.delta(eps, n_t, enc.pooled.float())
            xi = torch.randn(eps.shape, generator=noise_gen).to(device)
            z = eps + wmix * delta_s + sigma * xi
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, enc)))
            d_anchor = normalized_distance(z, eps)

        # ---- reward: the evaluation metric itself, plus the measured collapse floor
        # One detector query per (colour, shape), not per fixed-size chunk: a batch holds
        # eps_per_batch different pairs, so chunking blindly would query half the images with
        # the wrong shape word.
        res = [None] * len(u8)
        groups: dict[tuple[str, str], list[int]] = {}
        for i, key in enumerate(zip(cols, shps)):
            groups.setdefault(key, []).append(i)
        for (col, shp), idx in groups.items():
            out = count_images(counter, [u8[i] for i in idx], col, shp)
            for j, i in enumerate(idx):
                res[i] = out[j]
        counted = np.array([r.count for r in res], float)
        img_sd = np.array([u8[i].mean(axis=2).std() for i in range(len(u8))], float)
        tgt = np.array(counts, float)
        reward = -np.abs(counted - tgt) - args.collapse_penalty * np.maximum(0.0, args.sd_floor - img_sd) / args.sd_floor

        # ---- advantage against a per-count baseline, then normalised
        adv = np.zeros_like(reward)
        for i, n in enumerate(counts):
            b = baseline[n]
            adv[i] = reward[i] - (b if b is not None else reward[i])
            baseline[n] = reward[i] if b is None else args.baseline_ema * b + (1 - args.baseline_ema) * reward[i]
        s = adv.std()
        adv = adv / (s + 1e-6) if s > 1e-6 else adv
        a_t = torch.tensor(adv, device=device, dtype=torch.float32)

        # ---- REINFORCE surrogate: -(A * d(log pi)/d(phi)), gradient only through `delta`
        delta = sampler.delta(eps, n_t, enc.pooled.float())
        score = (xi * delta).flatten(1).sum(1) * (wmix / sigma)
        loss = -(a_t * score).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(sampler.parameters(), float(cfg.train.get("grad_clip", 10.0)))
        opt.step()

        if step % args.log_every == 0:
            rec = {"step": step, "reward": float(reward.mean()), "abs_err": float(np.abs(counted - tgt).mean()),
                   "counted_mean": float(counted.mean()), "target_mean": float(tgt.mean()),
                   "img_sd": float(img_sd.mean()), "collapsed_frac": float((img_sd < args.sd_floor).mean()),
                   "geom/d_anchor_mean": float(d_anchor.mean()), "geom/d_anchor_max": float(d_anchor.max()),
                   "grad_norm": float(gnorm), "loss": float(loss.detach()), "sec_per_step": (time.time() - t0) / step}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{step:6d}] reward {rec['reward']:+7.3f}  |counted-n| {rec['abs_err']:6.3f}  "
                  f"counted {rec['counted_mean']:5.2f} vs {rec['target_mean']:5.2f}  "
                  f"d_anchor {rec['geom/d_anchor_mean']:.4f}  img_sd {rec['img_sd']:5.1f}  "
                  f"collapsed {100*rec['collapsed_frac']:3.0f}%  {rec['sec_per_step']:.2f}s/step", flush=True)

        if args.preview_every and step % args.preview_every == 0:
            with torch.no_grad():
                pe = torch.randn(1, c, h, w_, generator=torch.Generator().manual_seed(0)).to(device)
                ns = list(range(1, n_max + 1))
                pp = [prompts.prompt(n, colors[0], shapes[0])[0] for n in ns]
                penc = gen.encode_prompts(pp)
                pz = pe.expand(len(ns), -1, -1, -1).contiguous()
                pz = pz + wmix * sampler.delta(pz, torch.tensor(ns, device=device), penc.pooled.float())
                im = gen.decode(gen.generate_latents(pz, penc))
                save_image_grid(torch.nn.functional.interpolate(im, scale_factor=0.5, mode="bilinear",
                                                                antialias=True),
                                out_dir / "previews" / f"step_{step:06d}.png", nrow=len(ns))
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_sampler_checkpoint(out_dir / "sampler_last.pt", sampler, cfg, step,
                                    latent_channels=gen.latent_channels, text_dim=gen.pooled_dim)

    save_sampler_checkpoint(out_dir / "sampler_last.pt", sampler, cfg, args.steps,
                            latent_channels=gen.latent_channels, text_dim=gen.pooled_dim)
    log.close()
    print(f"done in {(time.time() - t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
