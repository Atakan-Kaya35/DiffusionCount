"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  METHOD A objective (spec §3.4, Eq. 15)                               ║
║  L_A = λ_cnt L_cnt + λ_gap L_gap + λ_anc L_anc + λ_pri L_pri [+ λ_ord L_ord]     ║
║  Schedule: stage 1 {cnt, anc, pri} → stage 2 + gap → stage 3 + ord               ║
║  (run g_log and g_lin as separate experiments). No external counter.             ║
║  Annealing of γ (sharpening) and T (temperature) per §3.3.                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
Config: configs/method_a.yaml → method.attention, method.spectral
"""
from __future__ import annotations

import torch

from ...generator.attention import (
    AttentionRequest,
    concat_self_features,
    mean_self_attention,
    object_attention_map,
)
from ...utils.misc import scheduled_value
from ..losses.spectral import SpectralParams, spectral_count_losses
from .base import CountMethod


class SpectralCountMethod(CountMethod):
    name = "A"
    TERMS = ("cnt", "gap")

    def __init__(self, cfg, generator):
        super().__init__(cfg, generator)
        a = cfg.method.attention
        self.self_layers = list(a.self_layers)
        self.cross_layers = list(a.cross_layers)
        self.smoothing_sigma = float(a.get("smoothing_sigma", 0.0))
        self.scfg = cfg.method.spectral
        self.affinity = self.scfg.get("affinity", "attention")
        if self.affinity not in ("attention", "feature_cosine"):
            raise ValueError("method.spectral.affinity must be 'attention' or 'feature_cosine'")

    def attention_request(self) -> AttentionRequest:
        return AttentionRequest(self_layers=self.self_layers, cross_layers=self.cross_layers,
                                capture_features=self.affinity == "feature_cosine")

    def params(self, step: int) -> SpectralParams:
        s = self.scfg
        return SpectralParams(
            K=int(s.K), tau_a=float(s.tau_a), gamma=scheduled_value(s.gamma, step), tau=float(s.tau),
            T=scheduled_value(s.T, step), tau_g=float(s.tau_g), ridge=float(s.ridge),
            ridge_mode=s.get("ridge_mode", "relative"), drop_self_loops=bool(s.get("drop_self_loops", False)),
            eps_k=float(s.eps_k), huber_eta=float(s.huber_eta), min_foreground=int(s.get("min_foreground", 8)))

    def method_terms(self, out, batch, step):
        gen = self.generator
        grid = gen.layer_grid(self.self_layers[0])
        tok = gen.token_indices(batch.prompts, batch.object_words)
        a_bar = object_attention_map(out.cross_attn, self.cross_layers, tok, gen.layer_grids(), grid,
                                     self.smoothing_sigma)
        p = self.params(step)
        feats = concat_self_features(out.self_feats, self.self_layers) if self.affinity == "feature_cosine" else None
        S = mean_self_attention(out.self_attn, self.self_layers) if feats is None else None
        res = spectral_count_losses(S, a_bar.flatten(1), batch.counts, p, feats=feats)
        with torch.no_grad():
            v = res.valid
            n = batch.counts.double()
            logs = {
                "A/k_hat_mean": res.k_hat.mean().item(),
                "A/k_hat_abs_err": (res.k_hat - n).abs()[v].mean().item() if v.any() else float("nan"),
                "A/lambda_next_mean": res.lambda_next.mean().item(),
                "A/n_fg_mean": res.n_fg.float().mean().item(),
                "A/valid_frac": v.float().mean().item(),
                "A/gamma": p.gamma,
                "A/T": p.T,
            }
        return {"cnt": res.l_cnt, "gap": res.l_gap}, logs
