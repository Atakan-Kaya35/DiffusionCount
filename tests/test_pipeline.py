"""End-to-end smoke tests on the tiny generator: training steps for A/B/C, counter training,
pre-checks, evaluation and checkpoint round trips."""
import json

import torch
from conftest import tiny_cfg

from countsampler.data import CounterSceneSampler, ShapeRenderer, build_latent_cache, counter_categories
from countsampler.evaluation import ColorComponentsCounter, run_evaluation, run_prechecks
from countsampler.novel.components import DensityCounter, freeze_counter, save_counter
from countsampler.novel.methods import build_method
from countsampler.training import SamplerTrainer, load_sampler_checkpoint, train_counter


def _tiny_counter(tmp_path, cfg, tiny_gen):
    cats = counter_categories(cfg.data.colors, list(cfg.data.train_shapes) + list(cfg.data.heldout_shapes))
    c = DensityCounter(in_channels=4, num_categories=len(cats), base_channels=8, channel_mult=(1, 2))
    meta = {"in_channels": 4, "categories": cats, "input_space": "latent", "image_size": None,
            "arch": {"base_channels": 8, "channel_mult": [1, 2]}}
    path = tmp_path / "counter.pt"
    save_counter(path, c, meta)
    return path


def _run_steps(cfg, tiny_gen, steps=3, **kw):
    trainer = SamplerTrainer(cfg, generator=tiny_gen, **kw)
    w0 = trainer.sampler.conv_out.weight.detach().clone()
    logs = [trainer.train_step() for _ in range(steps)]
    tiny_gen.unwatch()
    assert all(lg["skipped"] == 0 for lg in logs), logs
    assert all(torch.isfinite(torch.tensor(lg["loss/total"])) for lg in logs)
    assert not torch.equal(w0, trainer.sampler.conv_out.weight)
    return trainer, logs


def test_method_a_trains(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path, ["method.stages=[{start: 0, terms: [cnt, gap, anc, pri, ord]}]"])
    _, logs = _run_steps(cfg, tiny_gen)
    assert "A/k_hat_mean" in logs[-1] and logs[-1]["A/valid_frac"] > 0


def test_method_a_feature_cosine_and_all_counts_geometry(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path, ["method.spectral.affinity=feature_cosine", "geometry.over=all_counts",
                                               "geometry.ord_form=rank", "geometry.g=lin"])
    _run_steps(cfg, tiny_gen, steps=2)


def test_method_b_trains(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_b.yaml", tmp_path)
    cfg.method.density.counter_ckpt = str(_tiny_counter(tmp_path, cfg, tiny_gen))
    _, logs = _run_steps(cfg, tiny_gen)
    assert "B/stray_mass" in logs[-1]


def test_method_c_trains_two_step_unroll(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_c.yaml", tmp_path, ["generator.timesteps=[999, 499]"])
    _, logs = _run_steps(cfg, tiny_gen)
    assert any(k.startswith("C/frac_peaks_ge_n@t") for k in logs[-1])


def test_stage_schedule(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path)
    m = build_method(cfg, tiny_gen)
    assert m.active_terms(0) == ("cnt", "anc", "pri")
    assert "gap" in m.active_terms(2500) and "ord" not in m.active_terms(2500)
    assert "ord" in m.active_terms(10_000)


def test_checkpoint_roundtrip_and_resume(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_c.yaml", tmp_path)
    trainer, _ = _run_steps(cfg, tiny_gen, steps=2)
    path = tmp_path / "s.pt"
    trainer.save(path)
    sampler, ckpt = load_sampler_checkpoint(path)
    assert ckpt["step"] == 2
    for a, b in zip(sampler.state_dict().values(), trainer.sampler.state_dict().values()):
        assert torch.equal(a, b)


def test_fit_writes_previews_and_checkpoint(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_c.yaml", tmp_path, ["train.preview.every=2", "train.preview.counts=[1,2]",
                                               "train.preview.seeds=[0]", "train.steps=2",
                                               "train.preview.prompts=[[red, circle]]"])
    trainer = SamplerTrainer(cfg, generator=tiny_gen, out_dir=tmp_path / "run")
    trainer.fit()
    tiny_gen.unwatch()
    assert (tmp_path / "run" / "sampler_last.pt").exists()
    assert (tmp_path / "run" / "previews" / "step_000002_red_circle.png").exists()


def test_evaluation_runs(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path)
    trainer = SamplerTrainer(cfg, generator=tiny_gen)
    tiny_gen.unwatch()
    for sampler in (None, trainer.sampler):
        out = tmp_path / ("eval_base" if sampler is None else "eval")
        m = run_evaluation(cfg, tiny_gen, sampler, ColorComponentsCounter(), out, split="all", seeds=[0, 1],
                           counts=[1, 2], colors=["red"], save_images=True)
        assert "overall" in m and (out / "per_seed_hist.csv").exists()
        assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["overall"]["n"] == 2 * 2 * 3


def test_prechecks_run(tiny_gen, tmp_path):
    cfg = tiny_cfg("method_a.yaml", tmp_path, [
        "precheck.counts=[1,2]", "precheck.samples_per_count=2", "precheck.batch_size=4",
        "precheck.timesteps=[999,500]", "precheck.resolutions=[16]", "precheck.A.gammas=[1.0]",
        "precheck.A.taus=[0.1]", "precheck.A.Ts=[0.02]", "precheck.A.K=64", "precheck.C.thresholds=[0.3]",
        "precheck.C.smoothing_sigmas=[0.0]"])
    counter_path = _tiny_counter(tmp_path, cfg, tiny_gen)
    from countsampler.novel.components import load_counter

    counter, meta = load_counter(counter_path)
    res = run_prechecks(cfg, tiny_gen, ["A", "B", "C"], tmp_path / "pc", hard_counter=ColorComponentsCounter(),
                        counter=counter, counter_meta=meta)
    assert res["A"] and res["B"] and res["C"]
    assert (tmp_path / "pc" / "precheck_A_summary.csv").exists()


def test_counter_training_image_mode(tmp_path):
    cfg = tiny_cfg("counter.yaml", tmp_path, [
        "counter.input_space=image", "counter.image_size=64", "counter.dataset.train_size=16",
        "counter.dataset.val_size=8", "counter.train.epochs=1", "counter.train.batch_size=4",
        "counter.train.num_workers=0", "counter.arch.base_channels=8", "counter.arch.channel_mult=[1,2]"])
    metrics = train_counter(cfg, out_dir=tmp_path / "counter")
    assert "mae_all" in metrics and (tmp_path / "counter" / "counter_last.pt").exists()


def test_latent_cache_with_tiny_vae(tiny_gen, tmp_path):
    cats = counter_categories(["red"], ["circle"])
    smp = CounterSceneSampler(cats, n_max=3)
    path = tmp_path / "cache.pt"
    build_latent_cache(path, smp, ShapeRenderer(size=32, radius_range=(0.08, 0.12)), cats, num=4, seed=0,
                       encode_fn=tiny_gen.encode_images, batch_size=2, image_aug={"blur_sigma_max": 1.0})
    data = torch.load(path, weights_only=True)
    assert data["latents"].shape == (4, 4, 16, 16) and len(data["objects"]) == 4
