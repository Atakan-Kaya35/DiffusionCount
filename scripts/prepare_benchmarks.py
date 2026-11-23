"""Prepare the §8 benchmarks.

  * writes the D2D-style reconstructions to benchmarks/d2d_style_{small,multi,large}.jsonl
    (deterministic, committed to the repo so every teammate evaluates the same prompts);
  * downloads CoCoCount.json from the Make It Count repo to data_cache/benchmarks/ (not committed,
    because that repo states no licence).

    python scripts/prepare_benchmarks.py
    python scripts/prepare_benchmarks.py --no-download
"""
import argparse
import urllib.request
from pathlib import Path

from _common import ROOT  # noqa: F401  (puts the repo on sys.path)

from countsampler.data.benchmarks import COCOCOUNT_URL, DEFAULT_PATHS, build_d2d_style, load_cococount, save_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true", help="skip downloading CoCoCount")
    ap.add_argument("--count-format", default="word", choices=["word", "digit"])
    args = ap.parse_args()

    for kind in ("small", "multi", "large"):
        path = Path(DEFAULT_PATHS[f"d2d_{kind}"])
        items = build_d2d_style(kind, count_format=args.count_format)
        save_jsonl(items, path)
        print(f"wrote {len(items)} prompts to {path}")

    if not args.no_download:
        dest = Path(DEFAULT_PATHS["cococount"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {COCOCOUNT_URL}")
        urllib.request.urlretrieve(COCOCOUNT_URL, dest)
        items = load_cococount(dest)
        print(f"CoCoCount: {len(items)} records ({len({it.prompt for it in items})} distinct prompts) in {dest}")


if __name__ == "__main__":
    main()
