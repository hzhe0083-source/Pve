"""Does a trained null-space adapter leave the old skill's actions unchanged? CPU only. For each trained PlaceCube run:
the Pick expert's head reading (a) the shared last-level vector, (b) its own level (the frozen base trunk), against the
expert itself, on stored Pick states and on random states; and what the old head would see from the adapter's write
if the projection were removed."""
import glob

import torch

from wrwl.nsa import build_actor

torch.set_grad_enabled(False)
exp = build_actor(torch.load("runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt", map_location="cpu"), 42, 8)
st = torch.load("data/bc_pick_c1.pt", map_location="cpu")
cands = [v for v in (st.values() if isinstance(st, dict) else [st]) if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == 42]
S = cands[0].float() if cands else None
torch.manual_seed(0)
R = torch.randn(8192, 42) * 2
print("stored Pick states:", None if S is None else tuple(S.shape), "| random states: (8192, 42)")
print(f"{'run':18s} trunk==expert | max|action change|: last level (stored/random) | own level | |W_old null| | old head would see without projection")
for arm in ("nsa64", "nsa64np", "nsa64npr"):
    for f in sorted(glob.glob(f"runs/protect_np/runs/place_{arm}_s*/final_ckpt.pt")):
        new = build_actor(torch.load(f, map_location="cpu"), 42, 8)
        lv = new[0]
        base = lv.trunk
        same = all(torch.equal(base[i].weight, exp[i].weight) and torch.equal(base[i].bias, exp[i].bias) for i in (0, 2, 4))
        row = []
        for X in ([S] if S is not None else []) + [R]:
            row.append((exp[-1](lv(X)) - exp(X)).abs().max().item())
        own = (exp[-1](base(R)) - exp(R)).abs().max().item()
        wn = (exp[-1].weight @ lv.null).abs().max().item()
        X = S if S is not None else R
        raw = (lv.adapter(torch.cat([X, base[:4](X)], 1)) @ exp[-1].weight.T).abs().max().item()
        name = f.split("/")[-2]
        print(f"{name:18s} {str(same):13s} | {' / '.join(f'{v:.1e}' for v in row):27s} | {own:.1e}     | {wn:.1e}      | {raw:.2f}")
