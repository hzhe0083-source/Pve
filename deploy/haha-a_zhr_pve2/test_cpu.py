"""CPU-only checks of the patched wrwl/ppo.py and wrwl/nsa.py (run with PYTHONPATH=<patched root>, CUDA_VISIBLE_DEVICES=)."""
import importlib.util
import shlex
import subprocess
import sys

import torch
import torch.nn as nn
import tyro

from wrwl.nsa import add_level, build_actor, depth, NullSpaceTrunk
from wrwl.ppo import Args

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)

print("tyro", tyro.__version__, "torch", torch.__version__)
for argv, want in ((["--nsa_project=False"], False), ([], True), (["--nsa_project", "False"], False),
                   (["--nsa_project=True"], True)):
    check(f"parse {argv}", tyro.cli(Args, args=argv).nsa_project is want)
# every queued arm's full argv (written by run_protect.sh / run_chain.sh / the driver, one per line in argv.txt)
for line in open(sys.argv[1]):
    name, _, rest = line.partition("\t")
    try:
        a = tyro.cli(Args, args=shlex.split(rest))
        check(f"argv {name}", True, f"project={a.nsa_project} init={a.nsa_init} fz={a.nsa_freeze_adapter} "
              f"adapter={a.nullspace_adapter} protect_old={a.protect_old} coef={a.protect_coef}")
    except SystemExit as e:
        check(f"argv {name}", False, f"exit {e.code}")

# nsa init: every variant starts as the old policy; null = I only without projection; nsa64 and nsa64npr start equal
torch.manual_seed(0)
base = nn.Sequential(nn.Linear(42, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
head = nn.Linear(256, 8)
flat = nn.Sequential(*base, head)
x = torch.randn(1024, 42) * 2
ref = flat(x)
lv = {}
for proj, init in ((True, None), (False, None), (False, "rand"), (True, "zero")):
    torch.manual_seed(1)
    am = add_level(flat, 42, 64, proj, init)
    eye = torch.equal(am[0].null, torch.eye(256))
    gap = (am(x) - ref).abs().max().item()
    w2 = am[0].adapter[2].weight.abs().max().item()
    check(f"init project={proj} init={init}", gap <= 1e-5 and eye == (not proj),
          f"head move {gap:.1e}, null==I {eye}, |W2|max {w2:.3f}")
    lv[(proj, init)] = am
a, b = lv[(True, None)][0], lv[(False, "rand")][0]
check("nsa64 vs npr: same first layer", torch.equal(a.adapter[0].weight, b.adapter[0].weight))
check("nsa64 vs npr: same initial features", (a(x) - b(x)).abs().max().item() < 1e-5,
      f"{(a(x) - b(x)).abs().max().item():.1e}")
check("npr adapter write is nonzero (random, not zero)", (b(x) - base(x)).abs().max().item() > 0.1,
      f"{(b(x) - base(x)).abs().max().item():.3f}")
check("np (zero init) writes nothing at init", (lv[(False, None)][0](x) - base(x)).abs().max().item() == 0)
# second level on a prefix-sum level, as chain64np builds it (unchanged path)
torch.manual_seed(2)
c2 = add_level(lv[(False, None)], 42, 64, False)
check("chain np level 2 builds", depth(c2[0]) == 2 and torch.equal(c2[0].null, torch.eye(256)))

# old checkpoints still load to the same function under the new nsa.py
spec = importlib.util.spec_from_file_location("nsa_old", sys.argv[2])
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
for ck in sys.argv[3:]:
    sd = torch.load(ck, map_location="cpu")
    if "actor_mean.0.null" not in sd and not any(k.startswith("actor_mean.0.adapter") for k in sd):
        sd = sd.get("model", sd)
    n, o = build_actor(sd, 42, 8), old.build_actor(sd, 42, 8)
    check(f"load {ck.split('/')[-2]}", torch.equal(n(x), o(x)), f"depth {depth(n[0]) if isinstance(n[0], NullSpaceTrunk) else 0}")
print("ALL PASS" if ok else "SOMETHING FAILED")
