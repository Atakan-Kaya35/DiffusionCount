"""What a bounded perturbation of the initial noise can and cannot do to the frozen generator.

Three measurements that the spec leaves as open choices and that decide whether
z_n = eps + w*Dphi(eps, n, c_bar) can work at all. Each writes a JSON of per-image rows.

  --mode radius     Push eps a fixed normalised distance in a RANDOM direction and decode.
                    d(a,b) = ||a-b||/sqrt(D), so z = eps + s*sqrt(D)*u/||u|| lands exactly at s.
                    Measured: at the full geometry.delta = 0.3 budget the image is essentially
                    unchanged (mean luminance 171.8 -> 167.2, corr(counted, n) +0.698 -> +0.749).
                    Distance alone does not break the generator.

  --mode direction  The same radii along a TRAINED sampler's own direction, against a random one
                    on the same eps and prompts.
                    Measured: the learned direction stops drawing scenes past 0.08 while the
                    random one survives 0.30+ - a 4x asymmetry, the ordinary adversarial one. The
                    generator denoises isotropic perturbation back toward the data manifold and
                    gradient descent finds the subspace that does not get denoised away.
                    Consequence: ||z - eps|| is the wrong quantity for L_anc to constrain, and no
                    choice of delta fixes it, because any delta big enough to let the sampler act
                    is big enough to let it destroy the image.

  --mode leverage   How much count change is AVAILABLE per unit distance along a path that cannot
                    collapse. Takes two noises the frozen model turns into very different counts
                    for the same prompt and slerps between them: spherical interpolation holds
                    ||z|| constant, so L_pri is satisfied at every step and every point is as
                    Gaussian as its endpoints. This is the feasibility question for the premise -
                    if no usable count change exists inside the safe radius, no loss can help.

Examples:
    python scripts/probe_perturbation.py --mode radius
    python scripts/probe_perturbation.py --mode direction --ckpt runs/d_ord/sampler_step001000.pt
    python scripts/probe_perturbation.py --mode leverage
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from _common import ROOT, base_parser, config_from_args

from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import normalized_distance
from countsampler.training import load_sampler_checkpoint
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images

RADII = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30]
COMBOS = [("blue", "pentagon"), ("red", "cross"), ("yellow", "diamond")]
WHITE = 'data.templates=["{count} {color} {shape} on a white background"]'


def at_radius(delta: torch.Tensor, s: float, d: int) -> torch.Tensor:
    """Rescale a [B,C,H,W] perturbation so that d(eps+out, eps) = s for every sample."""
    nrm = delta.flatten(1).norm(dim=1).clamp_min(1e-9)
    return delta * (s * (d ** 0.5) / nrm)[:, None, None, None]


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation: the output stays on the sphere through a and b, so L_pri holds."""
    af, bf = a.flatten(), b.flatten()
    an, bn = af / af.norm(), bf / bf.norm()
    om = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7))
    s = torch.sin(om)
    return ((torch.sin((1 - t) * om) / s) * af + (torch.sin(t * om) / s) * bf).view_as(a)


def main() -> None:
    ap = base_parser(__doc__, config_required=False)
    ap.add_argument("--mode", required=True, choices=["radius", "direction", "leverage"])
    ap.add_argument("--ckpt", default=None, help="sampler checkpoint (direction mode)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--counts", default="1,2,4,8", help="counts per prompt (radius/direction)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "direction" and not args.ckpt:
        ap.error("--mode direction needs --ckpt")

    args.config = args.config or [str(ROOT / "configs/base.yaml"), str(ROOT / "configs/gpu_8gb.yaml")]
    args.set = list(args.set) + [WHITE]
    cfg = config_from_args(args)
    seed_everything(int(cfg.experiment.seed))
    gen = FrozenTurboGenerator.from_config(cfg.generator,
                                           resolve_device(cfg.experiment.get("device", "auto")))
    owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
    ps = PromptSampler(cfg.data)
    c, h, w = gen.latent_shape
    dim = c * h * w
    counts = [int(x) for x in args.counts.split(",")]

    def decode_count(z, enc, color, shape):
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, enc)))
        res = count_images(owl, list(u8), color, shape)
        return ([r.count for r in res],
                [float(u8[i].mean(axis=2).mean()) for i in range(len(u8))],
                [float(u8[i].mean(axis=2).std()) for i in range(len(u8))])

    rows: list[dict] = []
    sampler = None
    if args.mode == "direction":
        sampler, _ = load_sampler_checkpoint(args.ckpt, "cpu")
        sampler = sampler.to(gen.device).eval()

    if args.mode in ("radius", "direction"):
        for color, shape in COMBOS:
            for seed in range(args.seeds):
                g = torch.Generator(device="cpu").manual_seed(2000 + seed)
                eps1 = torch.randn(1, c, h, w, generator=g)
                u = torch.randn(1, c, h, w, generator=g).to(gen.device)
                eps1 = eps1.to(gen.device)
                eps = eps1.expand(len(counts), -1, -1, -1).contiguous()
                prompts = [ps.prompt(n, color, shape)[0] for n in counts]
                enc = gen.encode_prompts(prompts)
                vecs = {"random": u.expand(len(counts), -1, -1, -1).contiguous()}
                if sampler is not None:
                    with torch.no_grad():
                        vecs["learned"] = sampler.delta(
                            eps, torch.tensor(counts, device=gen.device), enc.pooled.float())
                for kind, vec in vecs.items():
                    for s in RADII:
                        z = eps + at_radius(vec, s, dim) if s > 0 else eps
                        cnt, lum, sd = decode_count(z, enc, color, shape)
                        for k, n in enumerate(counts):
                            rows.append(dict(kind=kind, color=color, shape=shape, seed=seed, n=n,
                                             s=s, counted=cnt[k], lum=lum[k], sd=sd[k]))
            print(f"  done {color} {shape}", flush=True)

        kinds = ["learned", "random"] if sampler is not None else ["random"]
        print("\n================ PERTURBATION RADIUS ================")
        for kind in kinds:
            print(f"\n  {kind} direction:")
            head = ("  {:>9} {:>11} {:>8} {:>12} {:>9} {:>7}  verdict"
                    .format("d(z,eps)", "mean count", "zero %", "corr(cnt,n)", "mean lum", "img sd"))
            print(head)
            for s in RADII:
                v = [r for r in rows if r["s"] == s and r["kind"] == kind]
                p = np.array([r["counted"] for r in v], float)
                n = np.array([r["n"] for r in v], float)
                cr = float(np.corrcoef(p, n)[0, 1]) if p.std() > 1e-9 else float("nan")
                lum = float(np.mean([r["lum"] for r in v]))
                sd = float(np.mean([r["sd"] for r in v]))
                dead = (p == 0).mean() > 0.5 or sd < 25
                verdict = "*** COLLAPSED ***" if dead else "scene"
                print("  {:9.2f} {:11.2f} {:7.1f}% {:+12.3f} {:9.1f} {:7.1f}  {}"
                      .format(s, p.mean(), 100 * (p == 0).mean(), cr, lum, sd, verdict))
            ok = []
            for s in RADII:
                v = [r for r in rows if r["s"] == s and r["kind"] == kind]
                if (np.mean([r["counted"] == 0 for r in v]) <= 0.5
                        and np.mean([r["sd"] for r in v]) >= 25):
                    ok.append(s)
            print("  largest radius still drawing a scene: {:.2f}".format(max(ok) if ok else 0.0))

    else:
        for color, shape in COMBOS:
            for n in (2, 5, 8):
                prompt = ps.prompt(n, color, shape)[0]
                g = torch.Generator(device="cpu").manual_seed(7000 + n)
                cand = torch.randn(12, c, h, w, generator=g).to(gen.device)
                enc12 = gen.encode_prompts([prompt] * 12)
                cc, _, _ = decode_count(cand, enc12, color, shape)
                lo, hi = int(np.argmin(cc)), int(np.argmax(cc))
                if cc[hi] - cc[lo] < 2:
                    print(f"  {color} {shape} n={n}: no spread among candidates; skipping", flush=True)
                    continue
                a, b = cand[lo:lo + 1], cand[hi:hi + 1]
                ts = torch.linspace(0, 1, 9)
                path = torch.cat([slerp(a, b, float(t)) for t in ts])
                d = normalized_distance(path, a).cpu().numpy()
                enc9 = gen.encode_prompts([prompt] * len(ts))
                pc, _, psd = decode_count(path, enc9, color, shape)
                for i in range(len(ts)):
                    rows.append(dict(color=color, shape=shape, n=n, t=float(ts[i]), d=float(d[i]),
                                     counted=pc[i], sd=psd[i], c_lo=cc[lo], c_hi=cc[hi]))
                print(f"  {color} {shape} n={n}: {cc[lo]} -> {cc[hi]} over d=0..{d[-1]:.2f}  "
                      f"path {pc}", flush=True)
        print("\n============ COUNT CHANGE AVAILABLE PER UNIT DISTANCE ============")
        print("  {:>11} {:>21} {:>8}  verdict".format("d(z,start)", "mean |count - start|", "img sd"))
        for lim in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00):
            v = [r for r in rows if 0 < r["d"] <= lim]
            if not v:
                continue
            near: dict = {}
            for r in v:
                near.setdefault((r["color"], r["shape"], r["n"]), []).append(r)
            best = [max(gp, key=lambda r: r["d"]) for gp in near.values()]
            dc = float(np.mean([abs(r["counted"] - r["c_lo"]) for r in best]))
            sd = float(np.mean([r["sd"] for r in best]))
            print("  <= {:7.2f} {:21.2f} {:8.1f}  {}"
                  .format(lim, dc, sd, "scene" if sd >= 25 else "*** COLLAPSED ***"))

    out = Path(args.out or ROOT / "runs" / f"probe_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\n  wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
