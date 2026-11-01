"""Do the smoke runs' new flags take effect? Reads smoke/runs/smoke_*/{args.json,final_ckpt.pt} (CPU only)."""
import json
import torch

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)

want = {"np": (False, None, False), "npr": (False, "rand", False), "fz": (True, None, True), "nsa": (True, None, False)}
for n, (proj, init, fz) in want.items():
    a = json.load(open(f"smoke/runs/smoke_{n}/args.json"))
    got = (a.get("nsa_project"), a.get("nsa_init"), a.get("nsa_freeze_adapter"))
    check(f"args smoke_{n}", got == (proj, init, fz) and a.get("nullspace_adapter") == 64, str(got))
    sd = torch.load(f"smoke/runs/smoke_{n}/final_ckpt.pt", map_location="cpu")
    null = sd["actor_mean.0.null"]
    eye = torch.equal(null, torch.eye(len(null)))
    check(f"null smoke_{n}", eye == (not proj), f"null==I {eye}")
    w1, b1 = sd["actor_mean.0.adapter.0.weight"], sd["actor_mean.0.adapter.0.bias"]
    w2, b2 = sd["actor_mean.0.adapter.2.weight"], sd["actor_mean.0.adapter.2.bias"]
    orth2 = (w2.T @ w2 - torch.eye(w2.shape[1])).abs().max().item()        # orthogonal init, gain 1: W2^T W2 = I
    orth1 = (w1 @ w1.T - 2 * torch.eye(w1.shape[0])).abs().max().item()    # gain sqrt(2): W1 W1^T = 2 I
    bmax = max(b1.abs().max().item(), b2.abs().max().item())
    print(f"     smoke_{n}: |W2^T W2 - I| {orth2:.1e}  |W1 W1^T - 2I| {orth1:.1e}  max|bias| {bmax:.1e}  max|W2| {w2.abs().max().item():.3f}")
    if n == "fz":
        check("fz adapter never moved (orthogonal init intact, biases exactly 0)", orth2 < 1e-5 and orth1 < 1e-5 and bmax == 0)
    elif n == "np":
        check("np last layer left zero init (it gets gradient at once)", w2.abs().max().item() > 0)
    else:
        check(f"{n} adapter trained (biases moved off 0)", bmax > 0)
    hb = sd["actor_mean.1.weight"]
    print(f"     smoke_{n}: head rows read the adapter's write? |W_head null| {(hb @ null).abs().max().item():.2e}")
print("ALL PASS" if ok else "SOMETHING FAILED")
