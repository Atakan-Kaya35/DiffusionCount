"""Minimal run logger: JSON-lines file + console, optional Weights & Biases."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, out_dir: str | Path, use_wandb: bool = False, project: str | None = None,
                 run_name: str | None = None, config: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_dir / "log.jsonl", "a", encoding="utf-8")
        self._t0 = time.time()
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(project=project or "count-sampler", name=run_name, config=config)
            except Exception as exc:  # wandb is optional
                print(f"[logger] wandb disabled: {exc}")

    def log(self, step: int, metrics: dict[str, Any], echo: bool = True) -> None:
        clean = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
        record = {"step": step, "elapsed_s": round(time.time() - self._t0, 1), **clean}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(clean, step=step)
        if echo:
            shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in clean.items())
            print(f"[{step:>6}] {shown}", flush=True)

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
