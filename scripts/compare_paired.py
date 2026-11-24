"""Paired comparison of two evaluations, matched on (colour, shape, seed, n).

`scripts/compare.py` compares marginal means, which on this evaluation are dominated by seed
variance: the frozen baseline's own MAE moves from 4.44 to 5.32 between its first and last eight
seeds, and the n>=4 response slope from +0.42 to +0.13. An unpaired difference of that size proves
nothing.

Every run here is evaluated on the same seeds, prompts and noise, so the comparison can be paired
and the seed variance cancels:

    python scripts/compare_paired.py runs/eval_baseline_owl runs/c_tuned_s0/eval

Reports the mean change in absolute count error per matched image, a bootstrap confidence interval
over seeds (the unit of resampling, since records from one seed are not independent), and the same
split by requested count.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401  (repo on sys.path, stdout encoding)

KEY = ("color", "shape", "seed", "n")


def _load(d: Path) -> dict[tuple, int]:
    path = d / "records.csv"
    if not path.exists():
        raise SystemExit(f"no records.csv in {d}")
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[tuple(row[k] for k in KEY)] = int(float(row["counted"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _zero_rates(ref: dict, cand: dict, shared: list) -> str:
    """Share of images the counter finds nothing in.

    A jump here means the detector stopped recognising the object at all, rather than counting it
    wrongly - the signature of a run that drifted in colour, shape or style away from the prompt.
    """
    r0 = np.mean([ref[k] == 0 for k in shared])
    c0 = np.mean([cand[k] == 0 for k in shared])
    out = [f"\n**Zero-detection rate** (counter found nothing)\n",
           f"  reference: {100 * r0:.1f}%   candidate: {100 * c0:.1f}%   "
           f"change: {100 * (c0 - r0):+.1f} points\n"]
    by_color: dict[str, list] = {}
    for k in shared:
        by_color.setdefault(k[0], []).append(k)
    out.append("\n| colour | ref zero % | cand zero % | ref mean | cand mean |")
    out.append("\n|---|---|---|---|---|")
    for col in sorted(by_color):
        ks = by_color[col]
        out.append(f"\n| {col} | {100 * np.mean([ref[k] == 0 for k in ks]):.1f} | "
                   f"{100 * np.mean([cand[k] == 0 for k in ks]):.1f} | "
                   f"{np.mean([ref[k] for k in ks]):.2f} | "
                   f"{np.mean([cand[k] for k in ks]):.2f} |")
    return "".join(out) + "\n"


def _boot(per_seed: dict[str, list[float]], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample whole seeds: records sharing a seed share a noise vector and are not independent."""
    seeds = sorted(per_seed)
    if len(seeds) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        vals = [v for i in pick for v in per_seed[seeds[i]]]
        means.append(float(np.mean(vals)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="baseline evaluation directory")
    ap.add_argument("candidate", help="trained-run evaluation directory")
    args = ap.parse_args()

    ref, cand = _load(Path(args.reference)), _load(Path(args.candidate))
    shared = sorted(set(ref) & set(cand))
    if not shared:
        raise SystemExit("no matching (colour, shape, seed, n) records")
    print(f"reference: {args.reference}  ({len(ref)} records)")
    print(f"candidate: {args.candidate}  ({len(cand)} records)")
    print(f"matched:   {len(shared)}\n")

    rows, per_seed, by_n = [], {}, {}
    for k in shared:
        n = int(k[3])
        d = abs(cand[k] - n) - abs(ref[k] - n)        # negative = candidate is closer to the truth
        rows.append(d)
        per_seed.setdefault(k[2], []).append(d)
        by_n.setdefault(n, []).append(d)
    arr = np.asarray(rows, dtype=float)
    lo, hi = _boot(per_seed)

    print("**Paired change in absolute count error** (negative = the candidate is better)\n")
    print(f"  mean change: {arr.mean():+.4f}")
    print(f"  95% CI (bootstrap over {len(per_seed)} seeds): [{lo:+.4f}, {hi:+.4f}]")
    verdict = ("candidate better" if hi < 0 else
               "candidate worse" if lo > 0 else
               "no significant difference")
    print(f"  verdict: **{verdict}**")

    # A lower MAE is not a better model. The frozen generator over-counts, so a sampler that
    # produces almost nothing scores well on MAE while carrying no count information at all.
    # Method C did exactly this: MAE 8.166 -> 5.311, "candidate better", while emitting 0.19
    # objects per image and dropping the response slope from +1.031 to +0.023.
    cm = np.mean([cand[k] for k in shared])
    rm = np.mean([ref[k] for k in shared])
    czero = np.mean([cand[k] == 0 for k in shared])
    t = np.asarray([int(k[3]) for k in shared], float)
    c = np.asarray([cand[k] for k in shared], float)
    hi_n = t >= 4
    slope = np.polyfit(t[hi_n], c[hi_n], 1)[0] if hi_n.sum() > 2 else float("nan")
    warn = []
    if czero > 0.4:
        warn.append(f"{100 * czero:.0f}% of images contain nothing the counter can find")
    if cm < 0.5 * t.mean():
        warn.append(f"counted mean {cm:.2f} against a requested mean of {t.mean():.2f}")
    if slope < 0.2:
        warn.append(f"response slope {slope:+.3f} - the count is no longer followed")
    if warn:
        print()
        print("  *** DEGENERATE: a lower error here does not mean a better model ***")
        for w in warn:
            print(f"    - {w}")
        print(f"    reference emitted {rm:.2f} objects per image, candidate {cm:.2f}")
    print(f"  improved on {100 * (arr < 0).mean():.1f}% of images, "
          f"worse on {100 * (arr > 0).mean():.1f}%, unchanged on {100 * (arr == 0).mean():.1f}%\n")

    print("| requested n | matched | mean change | better % | worse % |")
    print("|---|---|---|---|---|")
    for n in sorted(by_n):
        v = np.asarray(by_n[n], dtype=float)
        print(f"| {n} | {len(v)} | {v.mean():+.3f} | {100 * (v < 0).mean():.1f} | "
              f"{100 * (v > 0).mean():.1f} |")

    print(_zero_rates(ref, cand, shared))

    # Response slope: the quantity that separates "counts better" from "found a better constant".
    t = np.asarray([int(k[3]) for k in shared], dtype=float)
    for label, src in (("reference", ref), ("candidate", cand)):
        p = np.asarray([src[k] for k in shared], dtype=float)
        hi = t >= 4
        print(f"  {label}: slope(all)={np.polyfit(t, p, 1)[0]:+.3f}  "
              f"slope(n>=4)={np.polyfit(t[hi], p[hi], 1)[0]:+.3f}  "
              f"counted mean={p.mean():.2f}")


if __name__ == "__main__":
    main()
