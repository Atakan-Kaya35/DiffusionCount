"""The positive result, shown. Count absent from the prompt; only the noise differs."""
import json, os, sys
import numpy as np, torch
WT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/.claude/worktrees/cs485-gpu-setup-5bb57e"
sys.path.insert(0, WT); os.chdir(WT)
from countsampler.data.prompts import PromptSampler
from countsampler.evaluation import build_hard_counter
from countsampler.evaluation.hard_counters import count_images
from countsampler.generator import FrozenTurboGenerator
from countsampler.utils.config import load_config
from countsampler.utils.misc import resolve_device, seed_everything, to_uint8_images
from PIL import Image, ImageDraw
FLAT = 'data.templates=["a flat lay photo of {count} {color} {shape} on a white background"]'
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percell"
T = list(range(10, 30, 2)); BL = 0.75
def slerp(a, b, t):
    if t == 0.0: return a.clone()
    af, bf = a.flatten().float(), b.flatten().float()
    an, bn = af / af.norm(), bf / bf.norm()
    om = torch.acos((an*bn).sum().clamp(-1+1e-7, 1-1e-7)); s = torch.sin(om)
    return ((torch.sin((1-t)*om)/s)*af + (torch.sin(t*om)/s)*bf).view_as(a)
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(0)
gen = FrozenTurboGenerator.from_config(cfg.generator, resolve_device(cfg.experiment.get("device", "auto")))
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
ps = PromptSampler(cfg.data)
c, h, w = gen.latent_shape
pool = torch.randn(128, c, h, w, generator=torch.Generator().manual_seed(4242))
eps = torch.randn(4, c, h, w, generator=torch.Generator().manual_seed(90210))
refs = json.load(open(f"{OUT}/refs.json"))
CELLS = [("green", "diamond"), ("red", "diamond")]
S, PAD, LAB, HDR = 190, 6, 170, 36
img = Image.new("RGB", (LAB + len(T)*(S+PAD), HDR + 2*len(CELLS)*(S+PAD+20) + 10), "white")
d = ImageDraw.Draw(img)
d.text((LAB, 6), "The count never appears in the prompt - every column below uses the identical text. "
                 "Only the noise differs.", fill="black")
d.text((LAB, 20), "Held-out seed: the image shown was never the one counted when the reference was chosen.", fill=(90,90,90))
row = 0
for color, shape in CELLS:
    p = ps.prompt(1, color, shape)[0]
    rf = {int(k): v for k, v in refs[f"{color}_{shape}"].items()}
    for name, zl in (("baseline: one fixed noise", [eps[0]]*len(T)),
                     (f"per-cell reference, blend {BL}", [slerp(eps[0], pool[rf[t]], BL) for t in T])):
        z = torch.stack(zl).to(gen.device)
        with torch.no_grad():
            u8 = to_uint8_images(gen.decode(gen.generate_latents(z, gen.encode_prompts([p]*len(T)))))
        cs = [r.count for r in count_images(owl, list(u8), color, shape)]
        y = HDR + row*(S+PAD+20)
        d.text((8, y+S//2-6), f"{color} {shape}", fill=(60,60,60)); d.text((8, y+S//2+8), name, fill="black")
        for k, t in enumerate(T):
            x = LAB + k*(S+PAD)
            img.paste(Image.fromarray(u8[k]).resize((S, S), Image.LANCZOS), (x, y))
            if row == 0: d.text((x+S//2-22, 22), f"ask {t}", fill="black")
            d.text((x+4, y+S+3), f"counted {cs[k]}",
                   fill=(0,130,0) if abs(cs[k]-t) <= 3 else ((200,120,0) if abs(cs[k]-t) <= 6 else (190,0,0)))
        print(f"  {color} {shape} / {name}: {cs}", flush=True)
        row += 1
img.save(f"{OUT}/comparison.png"); print(f"  wrote {OUT}/comparison.png")
