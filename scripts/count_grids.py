"""Count a finished evaluation's saved grids with OWLv2, on the CPU.

Why this exists. `scripts/evaluate.py` runs the detector on the same 8 GB card as the generator
and the VAE, so a 1440-image evaluation spends most of its wall clock in OWLv2 and the GPU cannot
be doing anything else. Splitting the two lets the GPU generate and the CPU count:

    python scripts/evaluate.py --ckpt runs/b_ord/sampler_last.pt --counter color \
        --set eval.counter_kwargs={} --out-dir runs/b_ord/eval_gen
    python scripts/count_grids.py runs/b_ord/eval_gen runs/b_ord/eval

The first command generates and saves the grids; its colour-counter numbers are discarded. The
second produces the real records.csv while the GPU has already moved on to the next training run.

**Resolution matters and is easy to get wrong.** `evaluate.py` counts the full 512 px decode but
saves its grids at `eval.grid_scale` (0.5 by default), so counting the grids is NOT the same
measurement as the records.csv of the run that saved them. Compare only runs counted the same way:
either re-count both with this script, or set `eval.grid_scale=1.0` so the saved grids are the
images that were counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _common import ROOT  # noqa: F401

from countsampler.evaluation.hard_counters import _nms, detector_query
from countsampler.evaluation.metrics import nn_regularity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="evaluation directory holding grids/")
    ap.add_argument("out", help="where to write records.csv and metrics.json")
    ap.add_argument("--cell", type=int, default=260, help="grid cell pitch in pixels (image + pad)")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--containment", type=float, default=0.8,
                    help="1.01 disables the containment rule, leaving plain IoU suppression")
    ap.add_argument("--threads", type=int, default=0, help="0 = half the cores, to leave the GPU fed")
    args = ap.parse_args()

    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 8) // 2))
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    post = (getattr(proc, "post_process_grounded_object_detection", None)
            or proc.post_process_object_detection)

    grids = sorted(f for f in os.listdir(src / "grids") if f.endswith(".png"))
    if not grids:
        raise SystemExit(f"no grids in {src / 'grids'}")
    rows, t0 = [], time.perf_counter()
    for gi, fn in enumerate(grids):
        color, shape = fn[:-4].split("_")
        g = Image.open(src / "grids" / fn)
        n_cols, n_rows = g.size[0] // args.cell, g.size[1] // args.cell
        cells, meta = [], []
        for sd in range(n_rows):
            for j in range(n_cols):
                cells.append(g.crop((j * args.cell, sd * args.cell,
                                     (j + 1) * args.cell, (sd + 1) * args.cell)).convert("RGB"))
                meta.append((sd, j + 1))
        for s0 in range(0, len(cells), args.batch):
            chunk = cells[s0:s0 + args.batch]
            inp = proc(text=[[detector_query(None, shape)]] * len(chunk), images=chunk,
                       return_tensors="pt")
            with torch.no_grad():
                o = model(**inp)
            # OWLv2 pads to a square, so centres must normalise by the LONGER side
            sizes = torch.tensor([[max(c.size), max(c.size)] for c in chunk])
            for k, r in enumerate(post(o, threshold=args.threshold, target_sizes=sizes)):
                b, s = r["boxes"], r["scores"]
                keep = _nms(b, s, args.iou, containment=args.containment) if len(s) else []
                sd, n = meta[s0 + k]
                side = max(chunk[k].size)
                centers = [(float((b[i, 0] + b[i, 2]) / 2 / side),
                            float((b[i, 1] + b[i, 3]) / 2 / side)) for i in keep]
                rows.append({"color": color, "shape": shape, "heldout_shape": "True", "seed": sd,
                             "n": n, "counted": len(keep),
                             "regularity": nn_regularity(centers), "clip": "nan"})
        el = time.perf_counter() - t0
        print(f"[{gi + 1}/{len(grids)}] {fn}  {len(rows)} images  {el / 60:.1f} min elapsed, "
              f"~{el / (gi + 1) * (len(grids) - gi - 1) / 60:.0f} min left", flush=True)

    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    p = np.array([r["counted"] for r in rows], float)
    t = np.array([r["n"] for r in rows], float)
    ks = sorted(set(int(x) for x in t))
    bar = min(np.abs(t - k).mean() for k in range(min(ks), max(ks) + 1))
    mae = float(np.abs(p - t).mean())
    m = {"overall": {"n": len(p), "exact_acc": float((p == t).mean()),
                     "within1_acc": float((np.abs(p - t) <= 1).mean()), "mae": mae,
                     "rel_err": float((np.abs(p - t) / np.maximum(t, 1)).mean())},
         "note": f"counted on CPU from {src.name}/grids, threshold {args.threshold}, "
                 f"iou {args.iou}, containment {args.containment}, no colour in the query"}
    for k in ks:
        m["overall"][f"exact_acc@{k}"] = float((p[t == k] == k).mean())
    m["heldout_shapes"] = dict(m["overall"])
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    hi = t >= 4
    print(f"\nDONE  MAE {mae:.3f}  skill {(bar - mae) / bar:+.3f}  exact {(p == t).mean():.4f}  "
          f"slope(n>=4) {np.polyfit(t[hi], p[hi], 1)[0]:+.3f}  "
          f"spread {np.mean([p[t == k].std() for k in ks]):.2f}  mean {p.mean():.2f}  "
          f"zero {100 * (p == 0).mean():.1f}%")


if __name__ == "__main__":
    main()
