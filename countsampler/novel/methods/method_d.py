"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD D objective: connected-component count                        ║
║  L_D = λ_cc L_cc + λ_bin L_bin + λ_anc L_anc + λ_pri L_pri [+ λ_ord L_ord]       ║
║  Reads the same object cross-attention map as Method C, on the same layer.       ║
║                                                                                  ║
║  Why it exists: Method C reached every one of its own objectives over 6000 steps ║
║  while producing one object for every count, because all of its terms are        ║
║  invariant under merging. L_cc is built on the Euler characteristic, which is    ║
║  not: merging two objects lowers it by exactly one.                              ║
║                                                                                  ║
║  Not a cure for everything. This still reads ATTENTION, so n strokes that draw   ║
║  n attention components satisfy it as well as n objects do - the tally-mark      ║
║  regime Method C passed through at step 1000 remains reachable. Guarding that    ║
║  needs a term grounded in the decoded image, which is Method B's territory.      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
Config: configs/method_d.yaml → method.attention, method.components
"""
from __future__ import annotations

import torch

from ...generator.attention import AttentionRequest, object_attention_map
from ..losses.components import ComponentParams, component_losses
from .base import CountMethod


class ComponentCountMethod(CountMethod):
    name = "D"
    TERMS = ("cc", "bin", "area")

    def __init__(self, cfg, generator):
        super().__init__(cfg, generator)
        a = cfg.method.attention
        self.cross_layers = list(a.cross_layers)
        self.smoothing_sigma = float(a.get("smoothing_sigma", 0.0))
        grid = a.get("grid")
        self.grid = tuple(grid) if grid else None
        cc = cfg.method.components
        self.params = ComponentParams(
            tau_frac=float(cc.tau_frac), temperature=float(cc.temperature),
            min_area=float(cc.get('min_area', 3.0)),
            eps=float(cc.get("eps", 1e-6)),
            hard_threshold=float(cc.get("hard_threshold", 0.5)))

    def attention_request(self) -> AttentionRequest:
        return AttentionRequest(cross_layers=self.cross_layers)

    def method_terms(self, out, batch, step):
        gen = self.generator
        grid = self.grid or gen.layer_grid(self.cross_layers[0])
        tok = gen.token_indices(batch.prompts, batch.object_words)
        a_bar = object_attention_map(out.cross_attn, self.cross_layers, tok, gen.layer_grids(), grid,
                                     self.smoothing_sigma)
        res = component_losses(a_bar, batch.counts, self.params)
        with torch.no_grad():
            n = batch.counts.float()
            # chi_hard is the discrete control. Method C's discrete diagnostic disagreed with its
            # differentiable loss for 1400 steps and the diagnostic was right, so every soft term
            # here ships with its hard counterpart and the two are watched for divergence.
            logs = {
                "D/chi_soft": res.chi.mean().item(),
                "D/chi_hard": res.chi_hard.mean().item(),
                "D/chi_hard_minus_n": (res.chi_hard - n).mean().item(),
                "D/chi_soft_minus_hard": (res.chi - res.chi_hard).mean().item(),
                "D/occupancy": res.occupancy.mean().item(),
                "D/area_per_component": res.area_per_component.mean().item(),
            }
        return {"cc": res.l_cc, "bin": res.l_bin, "area": res.l_area}, logs
