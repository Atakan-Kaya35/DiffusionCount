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
    print(f"  improved on {100 * (arr < 0).mean():.1f}% of images, "
          f"worse on {100 * (arr > 0).mean():.1f}%, unchanged on {100 * (arr == 0).mean():.1f}%\n")

    print("| requested n | matched | mean change | better % | worse % |")
    print("|---|---|---|---|---|")
    for n in sorted(by_n):
        v = np.asarray(by_n[n], dtype=float)
        print(f"| {n} | {len(v)} | {v.mean():+.3f} | {100 * (v < 0).mean():.1f} | "
              f"{100 * (v > 0).mean():.1f} |")


if __name__ == "__main__":
    main()
