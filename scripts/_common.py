"""Shared CLI plumbing: repo on sys.path, `--config` (repeatable) and `--set key=value` overrides."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Diagnostics print spec symbols (k̂, γ, λ). On Windows stdout defaults to the locale
# codepage (cp1252), which cannot encode them and aborts the run mid-scan.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from countsampler.utils.config import load_config  # noqa: E402


def base_parser(description: str, config_required: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", action="append", required=config_required, default=None,
                    help="YAML config; repeat to overlay (later files win)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides, e.g. --set train.steps=200 method.spectral.K=128")
    return ap


def config_from_args(args):
    return load_config(args.config, args.set)
