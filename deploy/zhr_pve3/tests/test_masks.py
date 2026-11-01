"""CPU checks of the rule-F masks in the patched wrwl/nsa.py, on the real Pick expert and stored Pick states:
every existing construction is bit-identical to the deployed nsa.py; every arm starts as the old policy; nsa64, npr,
vis and none share the adapter's draw and the global random stream; the masks are N, I, Pi, 0; the no-write arm gets no
adapter gradient; the visible-only arm's write reaches the old head.
  python test_masks.py NEW_NSA OLD_NSA EXPERT STATES|random"""
import importlib.util
import sys

import torch

new_path, old_path, expert, states = sys.argv[1:5]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


new, old = load(new_path, "nsa_new"), load(old_path, "nsa_old")
base = new.build_actor(torch.load(expert, map_location="cpu"), 42, 8)
if states == "random":      # Gaussian states, wider than the data (the checks hold for every input)
    X = torch.randn(4096, 42, generator=torch.Generator().manual_seed(0)) * 2
else:
    st = torch.load(states, map_location="cpu")
    X = (st["x"] if isinstance(st, dict) and "x" in st else next(v for v in st.values() if torch.is_tensor(v)
         and v.dim() == 2 and v.shape[1] == 42)).float()[:4096]
ref = base(X).detach()
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        fails.append(name)


def build(mod, **kw):
    torch.manual_seed(1)
    level = mod.add_level(base, 42, 64, **kw)
    return level, torch.get_rng_state()


# 1. existing constructions: identical weights, buffers and global stream to the deployed nsa.py
for name, kw in (("nsa64", {}), ("npr", dict(project=False, init="rand")), ("np", dict(project=False)),
                 ("nsa64z", dict(init="zero"))):
    a, ra = build(new, **kw)
    b, rb = build(old, **kw)
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    check(f"{name} bit-identical to the deployed nsa.py", same and torch.equal(ra, rb))

# 2. the rule-F arms
arms = {"nsa64": {}, "npr": dict(project=False, init="rand"), "vis": dict(mask="visible"), "none": dict(mask="zero")}
L = {k: build(new, **kw) for k, kw in arms.items()}
eye = torch.eye(256)
N = L["nsa64"][0][0].null
for k, (lv, rng) in L.items():
    gap = (lv(X) - ref).abs().max().item()
    check(f"{k} starts as the old policy", gap <= 1e-5, f"max |action change| {gap:.1e}")
    check(f"{k} keeps nsa64's global random stream", torch.equal(rng, L["nsa64"][1]))
    check(f"{k} first adapter layer is nsa64's draw",
          torch.equal(lv[0].adapter[0].weight, L["nsa64"][0][0].adapter[0].weight))
want = {"nsa64": N, "npr": eye, "vis": eye - N, "none": torch.zeros(256, 256)}
for k, m in want.items():
    check(f"{k} mask", (L[k][0][0].null - m).abs().max().item() <= 1e-6)
check("npr, vis and none start from the same projected draw",
      torch.equal(L["npr"][0][0].adapter[2].weight, L["vis"][0][0].adapter[2].weight)
      and torch.equal(L["vis"][0][0].adapter[2].weight, L["none"][0][0].adapter[2].weight))

# 3. gradients and who reads the write: one SGD step on the adapter, through the new head
W_old = base[-1].weight.detach().clone()
for k in ("nsa64", "vis", "none"):
    lv = L[k][0]
    head = lv[1]
    head.weight.data.add_(0.05 * torch.randn_like(head.weight))     # the new head has moved off the old rows
    out = head(lv[0](X))
    loss = (out * torch.randn_like(out)).sum()
    grads = torch.autograd.grad(loss, list(lv[0].adapter.parameters()))
    gmax = max(g.abs().max().item() for g in grads)
    with torch.no_grad():
        for p, g in zip(lv[0].adapter.parameters(), grads):
            p.sub_(1e-2 * g / (g.abs().max() + 1e-12))
    seen = (lv[0](X) @ W_old.T - base[:-1](X) @ W_old.T).abs().max().item()     # what the old head reads now
    if k == "none":
        check("none: no adapter gradient", gmax == 0.0, f"max |grad| {gmax:.1e}")
        check("none: the old head reads nothing new", seen == 0.0, f"{seen:.1e}")
    elif k == "nsa64":
        check("nsa64: the old head still reads nothing new after a step", seen <= 1e-5, f"{seen:.1e}")
    else:
        check("vis: the adapter learns", gmax > 0, f"max |grad| {gmax:.1e}")
        check("vis: the old head reads the write after a step", seen > 1e-3, f"{seen:.1e}")
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
