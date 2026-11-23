"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD B objective (spec §4.3, Eq. 17)                               ║
║  L_B = λ_den L_den + λ_anc L_anc + λ_pri L_pri [+ λ_ord L_ord]                   ║
║  Schedule: stage 1 {den, anc, pri} → stage 2 + ord. No attention maps are used.  ║
║  An amortised variant of D2D, with a trained counter in place of a detector.     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
Prerequisite: a trained counter checkpoint (scripts/train_counter.py).
Config: configs/method_b.yaml → method.density
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..components.counter import load_counter
from ..losses.density import category_targets, density_count_loss, timestep_weight
from .base import CountMethod


def counter_input(generator, x0_hat: torch.Tensor, input_space: str, image_size: int) -> torch.Tensor:
    """x̂0 latents as-is (latent counter) or decoded and resized WITH gradient (image counter)."""
    if input_space == "latent":
        return x0_hat
    img = generator.decode(x0_hat, grad=True)
    if img.shape[-1] != image_size:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False,
                            antialias=True)
    return img


class DensityCountMethod(CountMethod):
    name = "B"
    TERMS = ("den",)

    def __init__(self, cfg, generator, counter=None, counter_meta=None):
        super().__init__(cfg, generator)
        d = cfg.method.density
        self.dcfg = d
        if counter is None:
            counter, counter_meta = load_counter(d.counter_ckpt, device=generator.device)
        self.counter = counter          # frozen: eval + requires_grad=False, used WITHOUT no_grad
        self.meta = counter_meta
        self.category_index = {c: i for i, c in enumerate(counter_meta["categories"])}
        self.input_space = counter_meta.get("input_space", "latent")
        self.image_size = int(counter_meta.get("image_size") or 256)  # None for latent counters
        for c in cfg.data.colors:
            for s in list(cfg.data.train_shapes) + list(cfg.data.heldout_shapes):
                if f"{c} {s}" not in self.category_index:
                    raise ValueError(f"counter has no channel for category '{c} {s}'")
        vae = generator.vae
        if self.input_space == "image" and vae is not None and hasattr(vae, "enable_gradient_checkpointing"):
            vae.enable_gradient_checkpointing()  # backprop through the VAE decoder is memory-heavy

    def method_terms(self, out, batch, step):
        x = counter_input(self.generator, out.x0_hat, self.input_space, self.image_size)
        n_hat = self.counter.count(x.float())                                        # [B, K]
        target = category_targets(self.category_index, batch.colors, batch.shapes, batch.counts)
        per = density_count_loss(n_hat, target, alpha=float(self.dcfg.alpha), beta=float(self.dcfg.beta),
                                 variant=self.dcfg.get("variant", "hinge"),
                                 tolerance=float(self.dcfg.get("tolerance", 0.0)),
                                 huber_delta=float(self.dcfg.get("huber_delta", 1.0)))
        w_t = timestep_weight(self.dcfg.get("timestep_weight", "constant"), out.t, out.alpha_bar,
                              int(self.dcfg.get("timestep_threshold", 500)))
        with torch.no_grad():
            on_target = (n_hat * (target > 0)).sum(-1)
            stray = (n_hat * (target == 0)).sum(-1)
            logs = {
                "B/count_abs_err": (on_target - batch.counts.float()).abs().mean().item(),
                "B/count_exact": (on_target.round() == batch.counts.float()).float().mean().item(),
                "B/stray_mass": stray.mean().item(),
            }
        return {"den": w_t * per.mean()}, logs
