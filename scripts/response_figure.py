"""The count-response figure: what each run actually draws against what was asked for.

A single MAE hides the two failure modes this project keeps meeting. A run that emits almost
nothing scores well on MAE (Method C: MAE 5.311 while producing 0.19 objects per image), and a
run that follows the count on one shape and inverts it on another averages to a flat line (the
frozen baseline: cross +1.91, pentagon -0.65). Both are obvious here and invisible in a table.

Draws, per run: the mean counted value against the requested count, a band at +-1 sd (the
"spread" the ceiling analysis says must roughly halve before skill can turn positive), the
identity line, and the best-constant predictor that any useful method has to beat.

    python scripts/response_figure.py runs/eval_baseline_white_s8 runs/d_ord/eval --out fig.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401


def load(d: Path):
    p, t, sh = [], [], []
    with open(d / "records.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p.append(float(r["counted"])); t.append(int(float(r["n"]))); sh.append(r["shape"])
    return np.array(p), np.array(t), np.array(sh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="runs/figures/response.png")
    ap.add_argument("--by-shape", action="store_true", help="one panel per shape as well")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for d in args.dirs:
        d = Path(d)
        try:
            runs.append((d.name if d.name != "eval" else d.parent.name, *load(d)))
        except OSError:
            print(f"(skipping {d}: no records.csv)")
    if not runs:
        raise SystemExit("nothing to plot")

    shapes = sorted({s for _, _, _, sh in runs for s in sh}) if args.by_shape else []
    panels = ["all"] + shapes
    fig, axes = plt.subplots(1, len(panels), figsize=(max(8.6, 5.4 * len(panels)), 5.4),
                             squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, panel in zip(axes[0], panels):
        ks = None
        for i, (name, p, t, sh) in enumerate(runs):
            m = np.ones(len(p), bool) if panel == "all" else (sh == panel)
            if m.sum() < 10:
                continue
            pp, tt = p[m], t[m]
            ks = sorted(set(int(x) for x in tt))
            mu = np.array([pp[tt == k].mean() for k in ks])
            sd = np.array([pp[tt == k].std() for k in ks])
            c = colors[i % len(colors)]
            ax.plot(ks, mu, "o-", color=c, lw=2, label=f"{name}  (spread {sd.mean():.2f})")
            ax.fill_between(ks, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        if ks:
            ax.plot(ks, ks, "k--", lw=1.4, label="perfect")
            bar_k = min(range(min(ks), max(ks) + 1),
                        key=lambda k: np.mean([abs(x - k) for x in ks]))
            ax.axhline(bar_k, color="0.45", ls=":", lw=1.4,
                       label=f"best constant (n={bar_k})")
        ax.set_xlabel("requested count n")
        ax.set_ylabel("counted objects")
        ax.set_title("all shapes" if panel == "all" else panel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Count response: band is +-1 sd at fixed n (the 'spread' that must halve)",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
