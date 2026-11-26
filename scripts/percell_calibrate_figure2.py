"""Show the newly calibrated cells, loading the checkpoints the calibration run saved.

Unlike percell_calibrate_figure.py this retrains nothing - percell_calibrate_more.py writes
sampler_<color>_<shape>.pt, so the pictures here are produced by exactly the weights that were
scored. Cells are taken from the command line, best-first.
"""
import json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.novel.shared import build_sampler
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal"
T = list(range(10, 30, 2))
CELLS = [tuple(a.split("/")) for a in sys.argv[1:]] or [("green","cross")]
SUB = json.load(open(f"{OUT}/summary.json"))
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed)); dev = resolve_device(cfg.experiment.get("device","auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
ps = PromptSampler(cfg.data); c,h,w = gen.latent_shape
eps = torch.randn(8, c, h, w, generator=torch.Generator().manual_seed(90210))
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
S,PAD,LAB,HDR = 190,6,185,40
img = Image.new("RGB",(LAB+len(T)*(S+PAD), HDR+2*len(CELLS)*(S+PAD+20)+10),"white")
d = ImageDraw.Draw(img)
d.text((LAB,6),"Per-cell calibrated sampler, newly added cells.  The count never appears in the "
               "prompt - every column uses the identical text.", fill="black")
d.text((LAB,22),"Held-out seed 0, never used in training.  Weights loaded from the checkpoints that "
                "were scored, not retrained.", fill=(90,90,90))
row = 0
for color, shape in CELLS:
    p = ps.prompt(1,color,shape)[0]; cb = gen.encode_prompts([p]).pooled[0].float()
    net = build_sampler(scfg, c, cb.shape[0]).to(dev)
    net.load_state_dict(torch.load(f"{OUT}/sampler_{color}_{shape}.pt", map_location=dev))
    net.eval(); NNd = torch.tensor(T, device=dev)
    sk = SUB[f"{color}/{shape}"]["skill"]
    for name in ("baseline: untouched noise", f"calibrated Dphi (skill {sk:+.3f})"):
        e = torch.stack([eps[0]]*len(T)).to(dev)
        if name.startswith("calibrated"):
            with torch.no_grad(): z = net(e, NNd, cb[None].expand(len(T),-1))
        else: z = e
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(T)))))
        cs = [r.count for r in count_images(owl, list(u8), color, shape)]
        y = HDR + row*(S+PAD+20)
        d.text((8,y+S//2-6), f"{color} {shape}", fill=(60,60,60)); d.text((8,y+S//2+8), name, fill="black")
        for k,t in enumerate(T):
            x = LAB + k*(S+PAD)
            img.paste(Image.fromarray(u8[k]).resize((S,S), Image.LANCZOS),(x,y))
            if row == 0: d.text((x+S//2-22,26), f"ask {t}", fill="black")
            d.text((x+4,y+S+3), f"counted {cs[k]}",
                   fill=(0,130,0) if abs(cs[k]-t)<=3 else ((200,120,0) if abs(cs[k]-t)<=6 else (190,0,0)))
        print(f"  {color}/{shape} / {name}: {cs}", flush=True)
        row += 1
img.save(f"{OUT}/comparison2.png"); print(f"  wrote {OUT}/comparison2.png")
