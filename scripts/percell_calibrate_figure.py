"""The per-cell calibrated sampler, shown on the two cells where the instrument is most reliable."""
import json, os, sys, time
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
GEN = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/general"
OUT = r"C:/Users/atakan/Documents/School/CS485/DiffusionCount/runs/percal"
T = list(range(10, 30, 2)); BL = 0.75; STEPS = 4000
CELLS = [("red","cross"), ("blue","cross")]
def slerp_b(a,b,t):
    af,bf=a.flatten(1).float(),b.flatten(1).float()
    an,bn=af/af.norm(dim=1,keepdim=True),bf/bf.norm(dim=1,keepdim=True)
    om=torch.acos((an*bn).sum(1,keepdim=True).clamp(-1+1e-7,1-1e-7)); s=torch.sin(om)
    return ((torch.sin((1-t)*om)/s)*af+(torch.sin(t*om)/s)*bf).view_as(a)
cfg = load_config([f"{WT}/configs/base.yaml", f"{WT}/configs/gpu_8gb.yaml"], [FLAT, "data.include_count=false"])
seed_everything(int(cfg.experiment.seed)); dev = resolve_device(cfg.experiment.get("device","auto"))
gen = FrozenTurboGenerator.from_config(cfg.generator, dev)
owl = build_hard_counter("owlv2", device=gen.device, threshold=0.1, query_color=False)
ps = PromptSampler(cfg.data); c,h,w = gen.latent_shape
pool = torch.load(f"{GEN}/pool.pt", weights_only=False)["pool"]
refs = json.load(open(f"{GEN}/refs.json"))
eps = torch.randn(8, c, h, w, generator=torch.Generator().manual_seed(90210))
scfg = dict(cfg.sampler); scfg["norm_preserve"] = True
S,PAD,LAB,HDR = 190,6,185,40
img = Image.new("RGB",(LAB+len(T)*(S+PAD), HDR+2*len(CELLS)*(S+PAD+20)+10),"white")
d = ImageDraw.Draw(img)
d.text((LAB,6),"Per-cell calibrated sampler  z = proj(eps + w*Dphi(eps, n, c_bar)).  The count never "
               "appears in the prompt - every column uses the identical text.", fill="black")
d.text((LAB,22),"Held-out seed, never used in training.  red/cross skill +0.360, blue/cross +0.280, "
                "both p < 0.0001.", fill=(90,90,90))
row = 0
for color, shape in CELLS:
    p = ps.prompt(1,color,shape)[0]; cb = gen.encode_prompts([p]).pooled[0].float()
    R = torch.stack([pool[refs[f"{color}_{shape}"][str(t)]] for t in T]).to(dev)
    NNd = torch.tensor(T, device=dev)
    net = build_sampler(scfg, c, cb.shape[0]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); g = torch.Generator().manual_seed(7)
    for step in range(STEPS):
        j = torch.randint(0,len(T),(16,),generator=g); e = torch.randn(16,c,h,w,generator=g).to(dev)
        loss = torch.nn.functional.mse_loss(net(e,NNd[j],cb[None].expand(16,-1)), slerp_b(e,R[j],BL))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
    net.eval()
    for name in ("baseline: untouched noise", "calibrated sampler Dphi"):
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
img.save(f"{OUT}/comparison.png"); print(f"  wrote {OUT}/comparison.png")
