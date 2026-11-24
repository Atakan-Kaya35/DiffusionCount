"""Docs drift checks: config reference completeness, relative links/anchors, and repository paths."""
import re
from pathlib import Path

import pytest

from countsampler.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
PATH_PREFIXES = ("countsampler/", "configs/", "scripts/", "tests/", "docs/", "benchmarks/")


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            yield from _flatten(v, key)
        else:
            yield key


def _slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*)$", md.read_text(encoding="utf-8"), flags=re.M)}


@pytest.mark.parametrize("cfg_path", sorted((ROOT / "configs").rglob("*.yaml")), ids=lambda p: p.name)
def test_every_config_key_is_documented(cfg_path):
    documented = set(re.findall(r"`([A-Za-z_][\w.]*)`", (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")))
    missing = []
    for key in _flatten(load_config(cfg_path)):
        parent = key.rsplit(".", 1)[0]
        if key not in documented and not (parent.count(".") >= 1 and parent in documented):
            missing.append(key)
    assert not missing, f"keys from {cfg_path.name} missing in docs/configuration.md: {missing}"


@pytest.mark.parametrize("md", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(md):
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (md.parent / file_part).resolve() if file_part else md
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(target)
    assert not broken, f"broken links in {md.name}: {broken}"


@pytest.mark.parametrize("md", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_backticked_repo_paths_exist(md):
    missing = []
    for token in re.findall(r"`([^`\s]+)`", md.read_text(encoding="utf-8")):
        path = token.split("::")[0]
        if path.startswith(PATH_PREFIXES) and not any(c in path for c in "*<>{}") and not (ROOT / path).exists():
            missing.append(token)
    assert not missing, f"paths in {md.name} that do not exist: {missing}"
