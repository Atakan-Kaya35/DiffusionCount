"""Full provenance + results report for one run directory (spec §8 bookkeeping).

`scripts/compare.py` prints the headline table across runs. This prints everything about *one*
run, so a result can be reproduced and audited later: environment, resolved config, checkpoint
hashes, training-curve summary per stage, and the complete evaluation metrics including the
per-count breakdown and a constant-predictor reference bar.

    python scripts/report_run.py runs/eval_baseline_owl
    python scripts/report_run.py runs/c_tuned_s0 --out docs/results/c_tuned_s0.md

An evaluation directory (one holding metrics.json) or a training run directory (one holding
log.jsonl, optionally with eval/ below it) are both accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401  (puts the repo on sys.path, fixes stdout encoding)

MISSING = "-"


def _fmt(v, nd: int = 4) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return MISSING if np.isnan(v) else f"{v:.{nd}f}"
    if isinstance(v, int):
        return str(v)
    return MISSING if v is None else str(v)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else MISSING
    except (OSError, subprocess.SubprocessError):
        return MISSING


def _sha256(path: Path, limit_mb: int = 512) -> str:
    """Hash the file. Checkpoints here are a few MB; the cap only guards against surprises."""
    if path.stat().st_size > limit_mb * 1024 * 1024:
        return f"(skipped, >{limit_mb} MB)"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- environment


def section_environment() -> str:
    lines = [f"- Commit: `{_git('rev-parse', '--short', 'HEAD')}` "
             f"on `{_git('rev-parse', '--abbrev-ref', 'HEAD')}`"]
    dirty = _git("status", "--porcelain")
    if dirty and dirty != MISSING:
        lines.append(f"- **Working tree dirty**: {len(dirty.splitlines())} modified file(s), so the "
                     f"commit above does not fully describe this run")
    lines.append(f"- Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    try:
        import torch

        lines.append(f"- torch {torch.__version__}, CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            lines.append(f"- GPU: {p.name}, {p.total_memory / 1e9:.1f} GB, "
                         f"compute capability sm_{p.major}{p.minor}")
    except ImportError:
        lines.append("- torch not importable in this interpreter")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- checkpoints


def section_checkpoints(run: Path) -> str:
    rows = []
    for c in sorted(run.glob("*.pt"), key=lambda p: p.stat().st_mtime):
        step, params = MISSING, MISSING
        try:
            import torch

            blob = torch.load(c, map_location="cpu", weights_only=False)
            if isinstance(blob, dict):
                step = str(blob.get("step", MISSING))
                sd = blob.get("sampler") or blob.get("model") or blob.get("state_dict")
                if isinstance(sd, dict):
                    params = f"{sum(v.numel() for v in sd.values() if hasattr(v, 'numel')):,}"
        except Exception as exc:  # noqa: BLE001 - a report must never fail on one bad file
            step = f"(unreadable: {type(exc).__name__})"
        rows.append([f"`{c.name}`", step, params, f"{c.stat().st_size / 1e6:.1f} MB",
                     f"`{_sha256(c)}`"])
    return _table(["file", "step", "parameters", "size", "sha256[:16]"], rows)


# --------------------------------------------------------------------------- training log


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def section_training(run: Path) -> str:
    log = run / "log.jsonl"
    if not log.exists():
        return ""
    rows = _read_jsonl(log)
    if not rows:
        return "_(log.jsonl is empty)_\n"

    elapsed = rows[-1].get("elapsed_s")
    out = [f"Logged lines: **{len(rows)}**, steps {rows[0].get('step')} to {rows[-1].get('step')}"
           + (f", wall time **{elapsed / 60:.1f} min**" if isinstance(elapsed, (int, float)) else "")
           + ".\n"]

    sec = [r["sec_per_step"] for r in rows if isinstance(r.get("sec_per_step"), (int, float))]
    mem = [r["gpu_mem_gb"] for r in rows if isinstance(r.get("gpu_mem_gb"), (int, float))]
    if sec:
        out.append(f"- `sec_per_step`: mean {np.mean(sec):.3f}, median {np.median(sec):.3f}\n")
    if mem:
        out.append(f"- peak `gpu_mem_gb`: {max(mem):.2f}\n")
    out.append(f"- skipped optimiser steps: **{sum(int(r.get('skipped') or 0) for r in rows)}** "
               f"of {len(rows)} logged; dropped micro-batches: "
               f"**{sum(int(r.get('skipped_micro') or 0) for r in rows)}**\n")

    keys = sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, (int, float)) and (k.startswith(("loss/", "geom/"))
                                                       or k.split("/")[0] in {"A", "B", "C"}
                                                       or k in {"grad_norm", "lr"})})
    by_stage: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        by_stage[r.get("stage", MISSING)].append(r)
    if keys:
        stages = sorted(by_stage.items(), key=lambda kv: str(kv[0]))
        header = ["key"] + [f"stage {s} (n={len(v)})" for s, v in stages]
        body = []
        for k in keys:
            cells = [f"`{k}`"]
            for _, rs in stages:
                vals = [r[k] for r in rs
                        if isinstance(r.get(k), (int, float)) and not np.isnan(r[k])]
                cells.append(f"{np.mean(vals):.4g}" if vals else MISSING)
            body.append(cells)
        out.append("\n**Mean of each logged scalar, by stage**\n\n" + _table(header, body))

        k = max(1, len(rows) // 10)
        head, tail = rows[:k], rows[-k:]
        moves = []
        for key in keys:
            a = [r[key] for r in head
                 if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
            b = [r[key] for r in tail
                 if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
            if a and b:
                moves.append([f"`{key}`", f"{np.mean(a):.4g}", f"{np.mean(b):.4g}",
                              f"{np.mean(b) - np.mean(a):+.4g}"])
        out.append(f"\n**First vs last {k} logged steps** (did the optimised quantity move?)\n\n"
                   + _table(["key", f"first {k}", f"last {k}", "change"], moves))
    return "".join(out)


# --------------------------------------------------------------------------- evaluation


def _records(eval_dir: Path) -> tuple[list[int], list[int]]:
    path = eval_dir / "records.csv"
    if not path.exists():
        return [], []
    pred, tgt = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                pred.append(int(float(row["counted"])))
                tgt.append(int(float(row["n"])))
            except (KeyError, TypeError, ValueError):
                continue
    return pred, tgt


def _constant_bar(tgt: list[int]) -> tuple[float, int]:
    """MAE of the best single constant prediction: the bar any real count signal must clear."""
    if not tgt:
        return float("nan"), 0
    t = np.asarray(tgt, dtype=float)
    best = min(range(int(t.min()), int(t.max()) + 1), key=lambda c: np.abs(t - c).mean())
    return float(np.abs(t - best).mean()), int(best)


def _response_curve(pred: list[int], tgt: list[int]) -> str:
    """Counted value as a function of the requested one.

    The single most diagnostic view of this project. A generator that ignores the prompt gives a
    flat row; one that counts correctly gives counted == requested; the frozen SD-Turbo baseline
    tracks the request at small n and then saturates. Mean error alone hides which of the three
    is happening, because a flat row and a saturating row can post the same MAE.
    """
    p, t = np.asarray(pred, float), np.asarray(tgt, float)
    rows = []
    for n in sorted(set(int(x) for x in t)):
        got = p[t == n]
        rows.append([str(n), f"{got.mean():.2f}", f"{np.median(got):.1f}",
                     f"{got.min():.0f}-{got.max():.0f}", f"{(got == n).mean():.3f}",
                     f"{got.mean() - n:+.2f}"])
    def _slope(mask) -> float:
        tt, pp = t[mask], p[mask]
        return float(np.polyfit(tt, pp, 1)[0]) if len(set(tt)) > 1 else float("nan")

    lo, hi = t <= 3, t >= 4
    note = ("\n**Count response.** Slope of counted against requested: "
            f"**{_slope(np.ones_like(t, bool)):+.3f}** overall "
            f"(1.0 = perfect tracking, 0.0 = prompt ignored).\n\n"
            "A single slope hides a saturating curve, so it is also split at n=3, which is where "
            "the frozen baseline stops following the prompt:\n\n"
            f"- n ≤ 3: **{_slope(lo):+.3f}**\n"
            f"- n ≥ 4: **{_slope(hi):+.3f}**\n\n"
            "A sampler that is really counting raises the second number. Raising only the first "
            "means it sharpened a range the frozen model already handled.\n\n")
    return note + _table(["requested n", "counted mean", "median", "range", "exact", "bias"], rows)


def section_evaluation(eval_dir: Path) -> str:
    mpath = eval_dir / "metrics.json"
    if not mpath.exists():
        return ""
    m = json.loads(mpath.read_text(encoding="utf-8"))
    out = [f"Source: `{eval_dir.as_posix()}`\n"]

    groups = [g for g in ("overall", "heldout_shapes", "train_shapes") if isinstance(m.get(g), dict)]
    if groups:
        rows = [[f"`{k}`"] + [_fmt(m[g].get(k)) for g in groups]
                for k in ("n", "exact_acc", "within1_acc", "mae", "rel_err")]
        out.append("\n**Count accuracy**\n\n" + _table(["metric"] + groups, rows))

        per_n = sorted({int(k.split("@")[1]) for g in groups for k in m[g]
                        if k.startswith("exact_acc@")})
        if per_n:
            rows = [[f"n={n}"] + [_fmt(m[g].get(f"exact_acc@{n}")) for g in groups] for n in per_n]
            out.append("\n**Exact accuracy by requested count**\n\n"
                       + _table(["requested"] + groups, rows))

    scalars = [(k, v) for k, v in m.items() if not isinstance(v, dict)]
    if scalars:
        out.append("\n**Run-level metrics**\n\n"
                   + _table(["metric", "value"], [[f"`{k}`", _fmt(v)] for k, v in scalars]))

    pred, tgt = _records(eval_dir)
    if pred:
        out.append(_response_curve(pred, tgt))
        bar, const = _constant_bar(tgt)
        mae = float(np.abs(np.asarray(pred, float) - np.asarray(tgt, float)).mean())
        skill = (bar - mae) / bar if bar else float("nan")
        r = float(np.corrcoef(pred, tgt)[0, 1]) if len(set(pred)) > 1 else float("nan")
        out.append(
            "\n**Is the count signal real?** A model that ignores the prompt can still post a "
            "respectable MAE, so the reference below is the best constant prediction on this very "
            "set. Skill is the fraction of that bar's error removed; 0 means no better than "
            "guessing one number for every prompt.\n\n"
            + _table(["quantity", "value"], [
                ["records", str(len(pred))],
                ["counted mean / median / max",
                 f"{np.mean(pred):.2f} / {np.median(pred):.1f} / {int(np.max(pred))}"],
                ["requested mean / median / max",
                 f"{np.mean(tgt):.2f} / {np.median(tgt):.1f} / {int(np.max(tgt))}"],
                ["MAE", f"{mae:.4f}"],
                [f"best-constant MAE (always answer {const})", f"{bar:.4f}"],
                ["**skill vs constant** (1 - MAE/bar)", f"**{skill:+.3f}**"],
                ["Pearson r(counted, requested)", _fmt(r, 3)],
            ]))
    return "".join(out)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory (a training run, or an evaluation directory)")
    ap.add_argument("--out", default=None, help="write markdown here instead of stdout")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f"not a directory: {run}")

    eval_dirs = [run] if (run / "metrics.json").exists() else []
    eval_dirs += sorted(d for d in run.glob("eval*") if (d / "metrics.json").exists())

    parts = [f"# {args.title or run.name}\n", "\n## Provenance\n\n" + section_environment()]

    cfg = run / "config.yaml"
    if cfg.exists():
        parts.append(f"\n## Resolved configuration\n\n`{cfg.as_posix()}`\n\n```yaml\n"
                     + cfg.read_text(encoding="utf-8").strip() + "\n```\n")

    ck = section_checkpoints(run)
    if "_(none)_" not in ck:
        parts.append("\n## Checkpoints\n\n" + ck)

    tr = section_training(run)
    if tr:
        parts.append("\n## Training\n\n" + tr)

    for d in eval_dirs:
        parts.append(f"\n## Evaluation - {d.name}\n\n" + section_evaluation(d))

    text = "".join(parts)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
