"""Sampler training loop (spec §7). Shared by all methods; the method object supplies the loss.

Each iteration:
  1. sample base noises ε and prompt templates;
  2. for each ε draw a set N of J counts (including the label count);
  3. z_n = ε + w Δφ(ε, n, c̄) (Eq. 2); run the frozen generator for the configured steps;
  4. method loss (A: attention + spectral, B: x̂0 + counter, C: cross-attention + soft centres);
  5. L_anc, L_pri (and later L_ord) across the J noises of each ε;
  6. update φ with gradient clipping and a skip-on-NaN guard.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from ..data.prompts import PromptSampler, PromptSet
from ..generator.turbo import FrozenTurboGenerator
from ..novel.methods import FlatBatch, build_method
from ..novel.shared import build_sampler
from ..utils.config import save_config, to_dict
from ..utils.misc import ensure_dir, resolve_device, save_image_grid, seed_everything
from ..utils.run_logger import RunLogger
from .checkpoint import load_sampler_checkpoint, save_sampler_checkpoint


class SamplerTrainer:
    def __init__(self, cfg, generator: FrozenTurboGenerator | None = None, method=None,
                 out_dir: str | Path | None = None):
        self.cfg = cfg
        exp, t = cfg.experiment, cfg.train
        seed_everything(int(exp.seed))
        device = resolve_device(exp.get("device", "auto"))
        self.generator = generator or FrozenTurboGenerator.from_config(cfg.generator, device)
        self.device = self.generator.device
        self.method = method or build_method(cfg, self.generator)
        self.generator.watch(self.method.attention_request())

        self.sampler = build_sampler(cfg.sampler, self.generator.latent_channels,
                                     self.generator.pooled_dim).to(self.device)
        self.opt = torch.optim.AdamW(self.sampler.parameters(), lr=float(t.lr),
                                     weight_decay=float(t.get("weight_decay", 0.0)),
                                     betas=tuple(t.get("betas", (0.9, 0.999))))
        warmup = int(t.get("warmup_steps", 0))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup) if warmup > 0 else 1.0)

        self.prompts = PromptSampler(cfg.data, split="train", seed=int(exp.seed))
        self.J = int(t.counts_per_eps)
        self.eps_per_micro = int(t.eps_per_batch)
        self.accum = int(t.get("grad_accum", 1))
        self.timesteps = list(cfg.generator.get("timesteps", [999]))
        self.geometry_over = cfg.geometry.get("over", "sampled")
        self.noise_gen = torch.Generator().manual_seed(int(exp.seed) + 1)

        self.out_dir = ensure_dir(out_dir or Path(exp.output_dir) / exp.name)
        save_config(cfg, self.out_dir / "config.yaml")
        self.logger = RunLogger(self.out_dir, use_wandb=bool(exp.get("wandb", False)),
                                project=exp.get("wandb_project"), run_name=exp.name, config=to_dict(cfg))
        self.step = 0
        if t.get("resume"):
            self.load(t.resume)

    # ------------------------------------------------------------------ batch construction
    def _sample_eps(self, n: int) -> torch.Tensor:
        return torch.randn(n, *self.generator.latent_shape, generator=self.noise_gen).to(self.device)

    def _flat_batch(self, sets: list[PromptSet], counts_per_set: list[list[int]] | None = None) -> FlatBatch:
        prompts, words, colors, shapes, counts, eps_idx = [], [], [], [], [], []
        for i, ps in enumerate(sets):
            for n in (counts_per_set[i] if counts_per_set else ps.counts):
                p, w = self.prompts.prompt(n, ps.color, ps.shape, ps.template)
                prompts.append(p)
                words.append(w)
                colors.append(ps.color)
                shapes.append(ps.shape)
                counts.append(n)
                eps_idx.append(i)
        return FlatBatch(prompts=prompts, object_words=words, colors=colors, shapes=shapes,
                         counts=torch.tensor(counts, device=self.device, dtype=torch.long),
                         eps_index=torch.tensor(eps_idx, device=self.device, dtype=torch.long))

    def _apply_sampler(self, eps: torch.Tensor, batch: FlatBatch):
        enc = self.generator.encode_prompts(batch.prompts)
        z = self.sampler(eps[batch.eps_index], batch.counts, enc.pooled.float())
        return z, enc

    def _geometry_inputs(self, eps, sets, z, batch):
        be = len(sets)
        if self.geometry_over == "sampled":
            return z.view(be, self.J, *z.shape[1:]), batch.counts.view(be, self.J)
        n_max = int(self.cfg.data.n_max)
        all_counts = [list(range(1, n_max + 1)) for _ in sets]
        gb = self._flat_batch(sets, all_counts)
        z_all, _ = self._apply_sampler(eps, gb)
        return z_all.view(be, n_max, *z.shape[1:]), gb.counts.view(be, n_max)

    # ------------------------------------------------------------------ one optimisation step
    def train_step(self) -> dict:
        self.sampler.train()
        self.opt.zero_grad(set_to_none=True)
        logs: dict = {}
        skipped = 0
        t0 = time.time()
        for _ in range(self.accum):
            sets = [self.prompts.sample(self.J) for _ in range(self.eps_per_micro)]
            eps = self._sample_eps(len(sets))
            batch = self._flat_batch(sets)
            try:
                z, enc = self._apply_sampler(eps, batch)
                outs = self.generator.unroll(z, enc, self.timesteps, capture=self.method.needs_attention)
                z_geom, counts_geom = self._geometry_inputs(eps, sets, z, batch)
                loss, step_logs = self.method.compute(self.step, eps=eps, z_geom=z_geom, counts_geom=counts_geom,
                                                      batch=batch, outs=outs)
            except torch.linalg.LinAlgError as exc:
                print(f"[step {self.step}] eigendecomposition failed ({exc}); skipping micro-batch")
                skipped += 1
                continue
            if not torch.isfinite(loss) or not loss.requires_grad:
                skipped += 1
                continue
            (loss / self.accum).backward()
            for k, v in step_logs.items():
                logs[k] = logs.get(k, 0.0) + v / self.accum

        params = [p for p in self.sampler.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(self.cfg.train.grad_clip)) if params else None
        step_skipped = skipped == self.accum or grad_norm is None or not torch.isfinite(grad_norm)
        if step_skipped:
            self.opt.zero_grad(set_to_none=True)
        else:
            self.opt.step()
            self.lr_sched.step()
        logs["skipped"] = int(step_skipped)      # 1 = no optimiser update this step
        logs["skipped_micro"] = skipped           # micro-batches dropped (non-finite loss / eigh failure)
        logs["grad_norm"] = float(grad_norm) if grad_norm is not None else float("nan")
        logs["lr"] = self.opt.param_groups[0]["lr"]
        logs["sec_per_step"] = time.time() - t0
        if self.device.type == "cuda":
            logs["gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
        self.step += 1
        return logs

    # ------------------------------------------------------------------ loop
    def fit(self, steps: int | None = None) -> None:
        t = self.cfg.train
        total = int(steps if steps is not None else t.steps)
        log_every = int(t.get("log_every", 10))
        ckpt_every = int(t.get("ckpt_every", 1000))
        prev = t.get("preview", {}) or {}
        prev_every = int(prev.get("every", 0))
        if prev_every and self.step == 0:
            self.preview("step_000000")
        while self.step < total:
            logs = self.train_step()
            if self.step % log_every == 0 or self.step == 1:
                self.logger.log(self.step, logs)
            if prev_every and self.step % prev_every == 0:
                self.preview(f"step_{self.step:06d}")
            if ckpt_every and self.step % ckpt_every == 0:
                self.save(self.out_dir / f"sampler_step{self.step:06d}.pt")
        self.save(self.out_dir / "sampler_last.pt")
        self.logger.close()

    # ------------------------------------------------------------------ previews
    @torch.no_grad()
    def preview(self, tag: str) -> None:
        """Image grid per (colour, shape): rows = seeds × {baseline ε, sampler z_n}, columns = counts."""
        prev = self.cfg.train.get("preview", {}) or {}
        seeds = list(prev.get("seeds", [0, 1]))
        counts = list(prev.get("counts", list(range(1, int(self.cfg.data.n_max) + 1))))
        pairs = prev.get("prompts") or [[self.cfg.data.colors[0], self.cfg.data.train_shapes[0]]]
        self.sampler.eval()
        for color, shape in pairs:
            eps_rows, prompts, ns = [], [], []
            for s in seeds:
                e = torch.randn(1, *self.generator.latent_shape, generator=torch.Generator().manual_seed(int(s)))
                for _ in range(2):  # baseline row, sampler row
                    for n in counts:
                        eps_rows.append(e)
                        prompts.append(self.prompts.prompt(n, color, shape)[0])
                        ns.append(n)
            eps = torch.cat(eps_rows).to(self.device)
            n_t = torch.tensor(ns, device=self.device)
            enc = self.generator.encode_prompts(prompts)
            z = self.sampler(eps, n_t, enc.pooled.float())
            row = torch.arange(len(ns), device=self.device) // len(counts)
            z = torch.where((row % 2 == 0)[:, None, None, None], eps, z)  # even rows: baseline (z = ε)
            imgs = self.generator.decode(self.generator.generate_latents(z, enc, self.timesteps))
            imgs = torch.nn.functional.interpolate(imgs, scale_factor=0.5, mode="bilinear", antialias=True)
            save_image_grid(imgs, self.out_dir / "previews" / f"{tag}_{color}_{shape}.png", nrow=len(counts))
        self.sampler.train()

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str | Path) -> None:
        save_sampler_checkpoint(path, self.sampler, self.cfg, self.step,
                                latent_channels=self.generator.latent_channels, text_dim=self.generator.pooled_dim,
                                optimizer=self.opt, lr_scheduler=self.lr_sched)

    def load(self, path: str | Path) -> None:
        sampler, ckpt = load_sampler_checkpoint(path, self.device)
        self.sampler.load_state_dict(sampler.state_dict())
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            self.lr_sched.load_state_dict(ckpt["lr_scheduler"])
        self.step = int(ckpt.get("step", 0))
        print(f"[trainer] resumed from {path} at step {self.step}")
