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
from collections import defaultdict
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


def project(delta: torch.Tensor, radius: float) -> torch.Tensor:
    """Scale each perturbation so d(eps + out, eps) <= radius. Enforced, not merely penalised.

    A penalty does not work here. The count-free baseline draws about 15.7 objects when the mean
    request is 5.5, so emitting NOTHING scores an error of 5.5 against 10.2 for drawing normally:
    collapse is the higher-reward policy and a soft penalty just sets the price of it. Measured
    directly - with a collapse penalty of 5.0 the run reached d_anchor 0.16, image sd 7.4 and
    100 % collapsed images within 25 steps. The radius comes from the amplification measurement:
    a learned direction turns d_z into 10-16x as much d_x0, the decode dies at d_x0 1.17, so
    d_z 0.07 maps to about 1.0 and is the usable ceiling.
    """
    d = delta.flatten(1).norm(dim=1) / (delta[0].numel() ** 0.5)
    scale = (radius / d.clamp_min(1e-9)).clamp(max=1.0)
    return delta * scale[:, None, None, None]


def to_shell(z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Rescale z back onto the sphere of radius ||eps||.

    Why every earlier policy-gradient run here was searching the wrong set. The radius above was
    derived from how far an ADDITIVE direction can travel before the decode dies: d_z 0.07, since
    a learned direction amplifies 10-16x into x0 and x0 dies at 1.17. But slerp paths carry the
    count across the whole band at d_z 1.112 without ever collapsing, because they hold ||z||
    fixed. The collapse was never about distance - it was about leaving the Gaussian shell the
    U-Net was trained on. So the 0.05-0.07 budget was not a safety margin; it was a cage 14x
    smaller than the region where count control lives, and six runs searched inside it.

    With this projection the displacement cap can be raised to the measured working range. The
    policy gradient stays valid: the Gaussian policy is over a = eps + w*Delta + sigma*xi, and
    this is a deterministic map applied to that action before the environment sees it.
    """
    zf, ef = z.flatten(1), eps.flatten(1)
    scale = ef.norm(dim=1) / zf.norm(dim=1).clamp_min(1e-9)
    return z * scale.view(-1, *([1] * (z.dim() - 1)))


def main() -> None:
    ap = base_parser(__doc__, config_required=False)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--radius-max", type=float, default=0.07,
                    help="ceiling for the adaptive radius; 0.07 is the smallest that can reach n=1")
    ap.add_argument("--radius", type=float, default=0.05,
                    help="hard cap on d(z, eps); the measured safe z-budget is 0.07-0.097")
    ap.add_argument("--sigma", type=float, default=0.02,
                    help="exploration sd in normalised distance; d(z, eps+wD) equals this")
    ap.add_argument("--norm-preserve", action="store_true",
                    help="rescale z onto the ||eps|| shell after composing it; this is what lifts "
                         "the displacement cage, so --radius may then be set near the measured 1.112")
    ap.add_argument("--cell", default=None,
                    help="restrict training to one colour/shape, e.g. red/cross; cells do not "
                         "share a solution and spreading one Dphi over ten dilutes all of them")
    ap.add_argument("--targets", default=None,
                    help="lo:hi:step for the requested counts, e.g. 10:28:2. The count-free prompt "
                         "reaches about 8-35 objects, so targets below that are unreachable by any noise")
    ap.add_argument("--eps-per-batch", type=int, default=2)
    ap.add_argument("--counts-per-eps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--counter", default="owlv2")
    ap.add_argument("--sd-floor", type=float, default=0.25,
                    help="backdrop fraction below which the decode is treated as collapsed; valid images span 0.48-0.74 at every count and a collapsed one scores 0.000")
    ap.add_argument("--sd-ceil", type=float, default=None,
                    help="backdrop fraction ABOVE which the decode is treated as empty. The floor "
                         "alone is one-sided: it was derived from dead decodes reading 0.000, so a "
                         "blank white canvas reads 1.000 and passes as healthy while drawing "
                         "nothing. Measured valid images span 0.48-0.74, so 0.80 is the natural "
                         "ceiling. Without this the policy is absorbed into drawing nothing: every "
                         "sample then scores the same worst reward, the advantage is identically "
                         "zero, and no gradient points back out.")
    ap.add_argument("--collapse-penalty", type=float, default=150.0,
                    help="K in the quadratic image-quality penalty K*((floor-sd)/floor)^2")
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
    if args.cell:
        cc, ss = args.cell.split("/")
        colors, shapes = [cc], [ss]
    if args.targets:
        lo, hi, st = (int(v) for v in args.targets.split(":"))
        target_pool = list(range(lo, hi + 1, st))
    else:
        target_pool = list(range(1, n_max + 1))
    baseline = defaultdict(lambda: (None, None))   # per-count (mean, scale) EMAs
    radius = float(args.radius)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        eps_list, counts, prom, words, cols, shps = [], [], [], [], [], []
        for _ in range(args.eps_per_batch):
            e = torch.randn(1, c, h, w_, generator=noise_gen)
            col = colors[int(torch.randint(len(colors), (1,), generator=noise_gen))]
            shp = shapes[int(torch.randint(len(shapes), (1,), generator=noise_gen))]
            idx = torch.randperm(len(target_pool), generator=noise_gen)[:args.counts_per_eps]
            ns = [target_pool[int(i)] for i in idx]
            for n in ns:
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
            delta_s = project(sampler.delta(eps, n_t, enc.pooled.float()), radius)
            xi = torch.randn(eps.shape, generator=noise_gen).to(device)
            z = eps + wmix * delta_s + sigma * xi
            if args.norm_preserve:
                z = to_shell(z, eps)
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
        # Validity is measured by how much of the BACKDROP survives, not by image variance.
        # img_sd confounds validity with count: a correct single-object image scores 45.4 and a
        # fourteen-object one 63.2, so any floor that rejects a collapsed decode (21-35) also
        # rejects half the low-count images - measured, 49 % of images with <= 3 objects fell
        # below the floor of 45. The policy obeyed that incentive and kept adding objects.
        # The backdrop fraction does separate them: 0.000 for a collapsed decode against 0.48-0.74
        # across every valid count, so a floor at 0.25 sits well clear of the valid range and
        # catches only collapse.
        mn = u8.min(axis=3).astype(np.int16)
        mx = u8.max(axis=3).astype(np.int16)
        img_sd = ((mn > 170) & ((mx - mn) < 40)).mean(axis=(1, 2)).astype(float)
        tgt = np.array(counts, float)
        # A BARRIER, not a penalty. The count-free baseline draws 15.74 objects for requests
        # averaging 5.5, so any route to fewer objects is rewarded - including wrecking the image.
        # Measured: with a scaled penalty, a partial collapse to image sd 20 cost only 1.0 and
        # scored 2.00 against the baseline's 3.61, so degradation was the cheapest way to cut the
        # count and the policy took it (38-50 % collapsed by step 400). At low n the count error
        # is so large that no proportional penalty can outweigh it, so the floor has to be a wall.
        # sd_floor is 45 against a baseline of about 62: this rejects DAMAGE, not just death.
        # Steep but CONTINUOUS, not a flat barrier. A flat -50 below the floor has no gradient
        # once every sample is inside it: the advantage is identically zero and the policy cannot
        # climb out. Measured - at step 150 every sample read reward -50.000 with image sd 35 and
        # the run was stuck. Quadratic with K = 150 costs 16.7 at sd 30 against a best-case count
        # gain of 10.7 (cutting 15.7 objects to 5 at n = 1), so degradation never pays, while the
        # gradient always points back toward a healthy image.
        rel = np.abs(counted - tgt) / np.maximum(tgt, 1.0)
        short = np.maximum(0.0, args.sd_floor - img_sd) / args.sd_floor
        over = (np.maximum(0.0, img_sd - args.sd_ceil) / max(1e-6, 1.0 - args.sd_ceil)
                if args.sd_ceil is not None else 0.0)
        reward = -rel - float(args.collapse_penalty) * (short ** 2 + over ** 2)

        # ---- advantage against a per-count baseline, then normalised
        # Per-count mean AND scale. The reward scale differs enormously across counts - a
        # relative error of 23 is routine at n = 1 and 1.4 is typical at n = 10 - so dividing by
        # the batch standard deviation across mixed counts lets the low counts dominate every
        # update whatever the policy did. Both moments are tracked per count instead.
        adv = np.zeros_like(reward)
        for i, n in enumerate(counts):
            b, sc = baseline[n]
            centred = reward[i] - (b if b is not None else reward[i])
            adv[i] = centred / ((sc if sc else 1.0) + 1e-6)
            nb = reward[i] if b is None else args.baseline_ema * b + (1 - args.baseline_ema) * reward[i]
            nsc = abs(centred) if not sc else args.baseline_ema * sc + (1 - args.baseline_ema) * abs(centred)
            baseline[n] = (nb, nsc)
        adv = np.clip(adv, -5.0, 5.0)
        a_t = torch.tensor(adv, device=device, dtype=torch.float32)

        # ---- REINFORCE surrogate: -(A * d(log pi)/d(phi)), gradient only through `delta`
        delta = project(sampler.delta(eps, n_t, enc.pooled.float()), radius)
        score = (xi * delta).flatten(1).sum(1) * (wmix / sigma)
        loss = -(a_t * score).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(sampler.parameters(), float(cfg.train.get("grad_clip", 10.0)))
        opt.step()

        # ---- adaptive radius: ENFORCE image quality instead of pricing it.
        # A reward penalty cannot do this job. The advantage subtracts a per-count baseline that
        # follows the policy down, so when every sample is equally degraded the quality term is
        # absorbed into the baseline and only the count differences reach the gradient - measured
        # three times, each ending at 100% of images below the floor. Shrinking the radius when
        # the batch is unhealthy removes the option rather than taxing it, and the corridor here
        # is narrow enough (reaching n=1 needs d_x0 0.97, collapse starts at 1.17) that the radius
        # has to sit on the boundary rather than be chosen once.
        healthy = float(img_sd.mean()) >= args.sd_floor and (
            args.sd_ceil is None or float(img_sd.mean()) <= args.sd_ceil)
        radius = min(args.radius_max, radius * 1.01) if healthy else max(0.01, radius * 0.97)

        if step % args.log_every == 0:
            rec = {"step": step, "reward": float(reward.mean()), "abs_err": float(np.abs(counted - tgt).mean()),
                   "counted_mean": float(counted.mean()), "target_mean": float(tgt.mean()),
                   "img_sd": float(img_sd.mean()), "collapsed_frac": float((img_sd < args.sd_floor).mean()),
                   "empty_frac": float((img_sd > args.sd_ceil).mean()) if args.sd_ceil is not None else 0.0,
                   "geom/d_anchor_mean": float(d_anchor.mean()), "geom/d_anchor_max": float(d_anchor.max()),
                   "grad_norm": float(gnorm), "loss": float(loss.detach()), "radius": radius, "sec_per_step": (time.time() - t0) / step}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{step:6d}] reward {rec['reward']:+7.3f}  |counted-n| {rec['abs_err']:6.3f}  "
                  f"counted {rec['counted_mean']:5.2f} vs {rec['target_mean']:5.2f}  "
                  f"d_anchor {rec['geom/d_anchor_mean']:.4f}  img_sd {rec['img_sd']:5.1f}  "
                  f"dead {100*rec['collapsed_frac']:3.0f}%  blank {100*rec['empty_frac']:3.0f}%  r {radius:.4f}  {rec['sec_per_step']:.2f}s/step", flush=True)

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
