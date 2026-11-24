"""YAML configuration: `defaults:` inheritance, multi-file merge and dotted CLI overrides.

    cfg = load_config(["configs/method_a.yaml", "configs/sdxl_turbo.yaml"],
                      overrides=["train.steps=200", "method.spectral.K=128"])
    cfg.train.steps        # attribute access
    cfg.select("method.spectral.K", 256)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict with attribute access. Nested dicts are wrapped recursively by `load_config`."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def select(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: wrap(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [wrap(v) for v in obj]
    return obj


def to_dict(obj: Any) -> Any:
    """Convert a (nested) Config back to plain python containers (safe for yaml / torch.save)."""
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular `defaults:` chain at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = raw.pop("defaults", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, _load_raw(path.parent / parent, seen + (path,)))
    return deep_merge(merged, raw)


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"override '{text}' must look like key.sub=value")
    return key.strip(), yaml.safe_load(value)


def load_config(paths: str | Path | Iterable[str | Path], overrides: Iterable[str] = ()) -> Config:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, _load_raw(Path(p)))
    for ov in overrides or ():
        key, value = parse_override(ov)
        set_dotted(merged, key, value)
    return wrap(merged)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
