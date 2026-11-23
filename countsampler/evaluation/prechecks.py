"""Method-specific pre-checks on the frozen model before any training (spec §8). This is also the
offline scan that sets layer ℓ, timestep t and γ, τ, T, ... (spec §9).

Two data sources:
  synthetic  rendered scenes with known counts (including identical and touching shapes),
             VAE-encoded and noised to x_t = √ᾱ_t x0 + √(1−ᾱ_t) ε for every scanned timestep t;
  generated  one-step samples of the frozen generator from pure noise. The true count is
             what a hard counter finds in the decoded image (the prompt count is kept too).

  A: does the soft count k̂ (Eq. 12) track the true count? Scanned over layers, timesteps,
     γ, τ, T, for both the attention-map affinity and the cosine-of-features fallback.
  B: does the counter read x̂0 reliably (MAE, exact accuracy)?
  C: how often does ā have ≥ n separate peaks? Scanned over layers, smoothing, threshold.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.prompts import build_prompt
from ..data.shapes import ShapeRenderer, pil_to_tensor
from ..generator.attention import AttentionRequest, object_attention_map
from ..novel.losses.density import category_targets
from ..novel.losses.peak import local_maxima
from ..novel.losses.spectral import (
    SpectralParams,
    attention_affinity,
    feature_affinity,
    gather_submatrix,
    normalized_laplacian,
    select_foreground,
    soft_count,
)
from ..utils.misc import ensure_dir


# ----------------------------------------------------------------------------- data sources
def _sample_specs(cfg, pc):
    rng = random.Random(int(pc.get("seed", 0)))
    colors = list(cfg.data.colors)
    shapes = list(cfg.data.train_shapes) + list(cfg.data.heldout_shapes)
    specs = []
    for n in pc.counts:
        for i in range(int(pc.samples_per_count)):
            specs.append({"n": int(n), "color": rng.choice(colors), "shape": rng.choice(shapes),
                          "touching": rng.random() < float(pc.touching_frac),
                          "identical": rng.random() < float(pc.identical_frac), "seed": len(specs)})
    return specs


def _prompts(cfg, specs):
    tpl = cfg.data.templates[0]
    fmt = cfg.data.get("count_format", "word")
    pw = [build_prompt(tpl, s["n"], s["color"], s["shape"], fmt, True) for s in specs]
    return [p for p, _ in pw], [w for _, w in pw]


def synthetic_source(cfg, gen, pc):
    """Yields (t, StepOutput, meta) for rendered scenes noised to every scanned timestep."""
    renderer = ShapeRenderer(size=gen.resolution, radius_range=tuple(pc.get("radius_range", (0.045, 0.09))))
    specs = _sample_specs(cfg, pc)
    bs = int(pc.batch_size)
    for i in range(0, len(specs), bs):
        chunk = specs[i:i + bs]
        imgs = []
        for s in chunk:
            scene = renderer.render(random.Random(s["seed"]), [(s["color"], s["shape"], s["n"])],
                                    touching=0.6 if s["touching"] else 0.0, identical=s["identical"])
            imgs.append(pil_to_tensor(scene.image))
        x0 = gen.encode_images(torch.stack(imgs))
        prompts, words = _prompts(cfg, chunk)
        enc = gen.encode_prompts(prompts)
        noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(i)).to(gen.device)
        meta = {"specs": chunk, "prompts": prompts, "words": words,
                "n_true": torch.tensor([s["n"] for s in chunk], device=gen.device)}
        for t in pc.timesteps:
            ab = gen.alpha_bar(int(t))
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            yield int(t), gen.forward_at(x_t, int(t), enc, capture=True), meta


def generated_source(cfg, gen, pc, hard_counter):
    """Yields (t, StepOutput, meta) for one-step samples from pure noise; true count from a hard counter."""
    specs = _sample_specs(cfg, pc)
    bs = int(pc.batch_size)
    t = int(gen.default_timesteps[0])
    for i in range(0, len(specs), bs):
        chunk = specs[i:i + bs]
        prompts, words = _prompts(cfg, chunk)
        enc = gen.encode_prompts(prompts)
        z = torch.cat([torch.randn(1, *gen.latent_shape, generator=torch.Generator().manual_seed(s["seed"]))
                       for s in chunk]).to(gen.device)
        out = gen.unroll(z, enc, [t], capture=True)[0]
        imgs = gen.decode(out.x0_hat)
        n_true = [hard_counter.count(imgs[j], s["color"], s["shape"]).count for j, s in enumerate(chunk)]
        meta = {"specs": chunk, "prompts": prompts, "words": words,
                "n_true": torch.tensor(n_true, device=gen.device)}
        yield t, out, meta


# ----------------------------------------------------------------------------- helpers
def scan_layers(gen, resolutions) -> tuple[list[str], list[str]]:
    info = gen.attention_layers()
    sel = [n for n, li in info.items() if li.grid[0] in set(int(r) for r in resolutions)]
    return [n for n in sel if info[n].kind == "self"], [n for n in sel if info[n].kind == "cross"]


def paired_cross(self_name: str, info) -> str | None:
    if self_name.endswith("attn1"):
        cand = self_name[:-1] + "2"
        if cand in info and info[cand].kind == "cross":
            return cand
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarize(rows, keys, pred_key, true_key="n_true"):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for g, rs in groups.items():
        p = np.array([r[pred_key] for r in rs], dtype=float)
        t = np.array([r[true_key] for r in rs], dtype=float)
        valid = t > 0
        p, t = p[valid], t[valid]
        if len(t) == 0:
            continue
        corr = float(np.corrcoef(p, t)[0, 1]) if len(t) > 2 and p.std() > 0 and t.std() > 0 else float("nan")
        out.append({**dict(zip(keys, g)), "n": len(t), "mae": float(np.abs(p - t).mean()),
                    "exact": float((np.round(p) == t).mean()), "pearson": corr,
                    "mean_pred_minus_true": float((p - t).mean())})
    return sorted(out, key=lambda r: r["mae"])


def _print_top(title, rows, k=15):
    print(f"\n=== {title} (top {min(k, len(rows))} of {len(rows)} by MAE) ===")
    for r in rows[:k]:
        print("  " + "  ".join(f"{k_}={v:.3f}" if isinstance(v, float) else f"{k_}={v}" for k_, v in r.items()))


# ----------------------------------------------------------------------------- Method A
def precheck_a(cfg, gen, pc, out_dir: Path, sources) -> list[dict]:
    pa = pc.A
    info = gen.attention_layers()
    self_layers, _ = scan_layers(gen, pc.resolutions)
    pairs = [(s, paired_cross(s, info)) for s in self_layers if paired_cross(s, info)]
    gen.watch(AttentionRequest(self_layers=[s for s, _ in pairs], cross_layers=[c for _, c in pairs],
                               capture_features="feature_cosine" in pa.affinities))
    rows = []
    with torch.no_grad():
        for src_name, src in sources:
            for t, out, meta in src:
                tok = gen.token_indices(meta["prompts"], meta["words"])
                for s_name, c_name in pairs:
                    grid = info[s_name].grid
                    a_bar = object_attention_map(out.cross_attn, [c_name], tok, gen.layer_grids(), grid,
                                                 float(pa.smoothing_sigma))
                    idx, mask = select_foreground(a_bar.flatten(1), int(pa.K), float(pa.tau_a))
                    for aff in pa.affinities:
                        for gamma in pa.gammas:
                            if aff == "attention":
                                W = attention_affinity(gather_submatrix(out.self_attn[s_name], idx), mask,
                                                       float(gamma), bool(pa.drop_self_loops))
                            else:
                                f = out.self_feats[s_name]
                                f_F = f.gather(1, idx[:, :, None].expand(-1, -1, f.shape[-1]))
                                W = feature_affinity(f_F, mask, float(gamma), bool(pa.drop_self_loops))
                            lam = torch.linalg.eigvalsh(normalized_laplacian(W, mask, float(pa.ridge),
                                                                            pa.ridge_mode))
                            for tau in pa.taus:
                                for T in pa.Ts:
                                    k_hat = soft_count(lam, float(tau), float(T))
                                    for j, spec in enumerate(meta["specs"]):
                                        rows.append({"source": src_name, "t": t, "layer": s_name, "affinity": aff,
                                                     "gamma": float(gamma), "tau": float(tau), "T": float(T),
                                                     "n_prompt": spec["n"], "n_true": int(meta["n_true"][j]),
                                                     "touching": spec["touching"], "identical": spec["identical"],
                                                     "k_hat": float(k_hat[j]), "n_fg": int(mask[j].sum())})
    gen.unwatch()
    _write_csv(out_dir / "precheck_A_rows.csv", rows)
    summary = _summarize(rows, ["source", "t", "layer", "affinity", "gamma", "tau", "T"], "k_hat")
    _write_csv(out_dir / "precheck_A_summary.csv", summary)
    _print_top("Pre-check A: soft count k̂ vs true count", summary)
    return summary


# ----------------------------------------------------------------------------- Method C
def precheck_c(cfg, gen, pc, out_dir: Path, sources) -> list[dict]:
    pcc = pc.C
    _, cross_layers = scan_layers(gen, pc.resolutions)
    gen.watch(AttentionRequest(cross_layers=cross_layers))
    rows = []
    with torch.no_grad():
        for src_name, src in sources:
            for t, out, meta in src:
                tok = gen.token_indices(meta["prompts"], meta["words"])
                for c_name in cross_layers:
                    grid = gen.layer_grid(c_name)
                    for sigma in pcc.smoothing_sigmas:
                        a_bar = object_attention_map(out.cross_attn, [c_name], tok, gen.layer_grids(), grid,
                                                     float(sigma))
                        for thr in pcc.thresholds:
                            peaks = local_maxima(a_bar, float(thr)).flatten(1).sum(-1)
                            for j, spec in enumerate(meta["specs"]):
                                rows.append({"source": src_name, "t": t, "layer": c_name, "sigma": float(sigma),
                                             "threshold": float(thr), "n_prompt": spec["n"],
                                             "n_true": int(meta["n_true"][j]), "touching": spec["touching"],
                                             "num_peaks": int(peaks[j])})
    gen.unwatch()
    _write_csv(out_dir / "precheck_C_rows.csv", rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["t"], r["layer"], r["sigma"], r["threshold"])].append(r)
    summary = []
    for (src, t, layer, sigma, thr), rs in groups.items():
        rs = [r for r in rs if r["n_true"] > 0]
        if not rs:
            continue
        summary.append({"source": src, "t": t, "layer": layer, "sigma": sigma, "threshold": thr, "n": len(rs),
                        "frac_peaks_ge_n": float(np.mean([r["num_peaks"] >= r["n_true"] for r in rs])),
                        "frac_peaks_eq_n": float(np.mean([r["num_peaks"] == r["n_true"] for r in rs])),
                        "mae": float(np.mean([abs(r["num_peaks"] - r["n_true"]) for r in rs]))})
    summary.sort(key=lambda r: -r["frac_peaks_ge_n"])
    _write_csv(out_dir / "precheck_C_summary.csv", summary)
    _print_top("Pre-check C: ā has ≥ n separate peaks (sorted by frac_peaks_ge_n)", summary)
    return summary


# ----------------------------------------------------------------------------- Method B
def precheck_b(cfg, gen, pc, out_dir: Path, sources, counter, counter_meta) -> list[dict]:
    from ..novel.methods.method_b import counter_input

    gen.unwatch()
    cat_index = {c: i for i, c in enumerate(counter_meta["categories"])}
    space = counter_meta.get("input_space", "latent")
    size = int(counter_meta.get("image_size") or 256)
    rows = []
    with torch.no_grad():
        for src_name, src in sources:
            for t, out, meta in src:
                n_hat = counter.count(counter_input(gen, out.x0_hat, space, size).float())
                specs = meta["specs"]
                tgt = category_targets(cat_index, [s["color"] for s in specs], [s["shape"] for s in specs],
                                       meta["n_true"])
                on = (n_hat * (tgt > 0)).sum(-1)
                stray = (n_hat * (tgt == 0)).sum(-1)
                for j, s in enumerate(specs):
                    rows.append({"source": src_name, "t": t, "n_prompt": s["n"], "n_true": int(meta["n_true"][j]),
                                 "n_hat": float(on[j]), "stray": float(stray[j]), "touching": s["touching"]})
    _write_csv(out_dir / "precheck_B_rows.csv", rows)
    summary = _summarize(rows, ["source", "t"], "n_hat")
    _write_csv(out_dir / "precheck_B_summary.csv", summary)
    _print_top("Pre-check B: counter on x̂0 vs true count", summary)
    return summary


def run_prechecks(cfg, gen, methods: list[str], out_dir: str | Path, hard_counter=None, counter=None,
                  counter_meta=None) -> dict:
    pc = cfg.precheck
    out_dir = ensure_dir(out_dir)

    def sources():
        srcs = []
        if "synthetic" in pc.sources:
            srcs.append(("synthetic", synthetic_source(cfg, gen, pc)))
        if "generated" in pc.sources:
            if hard_counter is None:
                raise ValueError("the 'generated' source needs a hard counter")
            srcs.append(("generated", generated_source(cfg, gen, pc, hard_counter)))
        return srcs

    results = {}
    if "A" in methods:
        results["A"] = precheck_a(cfg, gen, pc, out_dir, sources())
    if "B" in methods:
        if counter is None:
            raise ValueError("pre-check B needs a trained counter (--counter-ckpt)")
        results["B"] = precheck_b(cfg, gen, pc, out_dir, sources(), counter, counter_meta)
    if "C" in methods:
        results["C"] = precheck_c(cfg, gen, pc, out_dir, sources())
    return results
