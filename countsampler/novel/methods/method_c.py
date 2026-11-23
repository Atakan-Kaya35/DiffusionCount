"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD C objective (spec §5, Eq. 22)                                 ║
║  L_peak = λ_act L_act + λ_rep L_rep + λ_supp L_supp                              ║
║  L_C = L_peak + λ_anc L_anc + λ_pri L_pri [+ λ_ord L_ord]                        ║
║  Schedule: stage 1 {act, supp, anc, pri} → stage 2 + rep → stage 3 + ord         ║
║  Uses only the object token's cross-attention ā (no self-attention, no counter). ║
╚══════════════════════════════════════════════════════════════════════════════════╝
Config: configs/method_c.yaml → method.attention, method.peak
"""
from __future__ import annotations

import torch

from ...generator.attention import AttentionRequest, object_attention_map
from ..losses.peak import PeakParams, peak_losses
from .base import CountMethod


class PeakSeparationMethod(CountMethod):
    name = "C"
    TERMS = ("act", "rep", "supp")

    def __init__(self, cfg, generator):
        super().__init__(cfg, generator)
        a = cfg.method.attention
        self.cross_layers = list(a.cross_layers)
        self.smoothing_sigma = float(a.get("smoothing_sigma", 0.0))
        grid = a.get("grid")
        self.grid = tuple(grid) if grid else None
        pc = cfg.method.peak
        self.params = PeakParams(
            kmeans_iters=int(pc.kmeans_iters), tau_q=float(pc.tau_q), tau_p=float(pc.tau_p),
            r_min=float(pc.r_min), r_s=float(pc.r_s), peak_min_value=float(pc.peak_min_value),
            supp_min_value=float(pc.get("supp_min_value", 0.0)), fps_min_value=float(pc.get("fps_min_value", 0.0)),
            eps=float(pc.get("eps", 1e-6)), diag_peak_threshold=float(pc.get("diag_peak_threshold", 0.3)))

    def attention_request(self) -> AttentionRequest:
        return AttentionRequest(cross_layers=self.cross_layers)

    def method_terms(self, out, batch, step):
        gen = self.generator
        grid = self.grid or gen.layer_grid(self.cross_layers[0])
        tok = gen.token_indices(batch.prompts, batch.object_words)
        a_bar = object_attention_map(out.cross_attn, self.cross_layers, tok, gen.layer_grids(), grid,
                                     self.smoothing_sigma)
        res = peak_losses(a_bar, batch.counts, self.params)
        with torch.no_grad():
            n = batch.counts.float()
            logs = {
                "C/num_peaks_minus_n": (res.num_peaks.float() - n).mean().item(),
                "C/frac_peaks_ge_n": (res.num_peaks.float() >= n).float().mean().item(),
                "C/peak_value_mean": (res.peak_values * res.center_valid).sum().item()
                / res.center_valid.sum().clamp_min(1).item(),
            }
        return {"act": res.l_act, "rep": res.l_rep, "supp": res.l_supp}, logs
