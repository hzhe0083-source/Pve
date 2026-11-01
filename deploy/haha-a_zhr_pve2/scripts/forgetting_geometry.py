"""Where does learning the new skill hurt the old one? Offline, from checkpoints already trained (no RL).

Old skill A = the PickCube expert; each checkpoint was warm-started from it and trained on the new skill. Each skill keeps
its own output head, so only the shared hidden layers are compared and A is always read through its OLD head. For each
hidden Linear l, with augmented input x~ = [x; 1] (bias included) on A's own states:
  C_l = E_A[x~ x~^T]          input directions A occupies
  Q_l = E_A[J_l^T J_l]        J_l = d mu_A / d z_l through A's downstream layers and old head: output directions A's
                              actions read
  P_in, P_out                 projectors on the top eigenvectors of C_l, Q_l holding --energy of their trace
For the actual update dW~_l = [W_new - W_old, b_new - b_old]:
  rho_l      ||P_out dW P_in||^2 / ||dW||^2   share of the new skill's update in the block that hurts A to first order
             (if high, blocking it would also block the new skill)
  gpm_l      ||dW P_in||^2 / ||dW||^2         share an input-only projection (GPM) would block
and A's action drift, in squared old-std units, three ways:
  D_true     E_A ||mu(new trunk, old head) - mu_A||^2                     exact
  D_1st      E_A ||sum_l J_l dW~_l x~_l||^2                                first order, per state, no factorisation
  D_fac      sum_l tr(dW~_l C_l dW~_l^T Q_l)                              Kronecker-factored, layer-decoupled
plus D_true after removing the blocked part from the update (ours: dW - P_out dW P_in; GPM: dW (I - P_in)).
  python scripts/forgetting_geometry.py --old=P --data=data/bc_pick_c1.pt --ckpts=a.pt,b.pt --out=logs/x.json
"""
import argparse
import json

import torch
from torch.func import jacrev, vmap

from wrwl.experts import load_policy

p = argparse.ArgumentParser()
p.add_argument("--old", required=True)
p.add_argument("--data", required=True, help="A's states, from conflict_concentration.py --collect=pick")
p.add_argument("--ckpts", required=True)
p.add_argument("--n", type=int, default=20000)
p.add_argument("--energy", default="0.9,0.95,0.99")
p.add_argument("--out", default="")
a = p.parse_args()
torch.manual_seed(0)
torch.set_num_threads(8)

old = load_policy(a.old, 42, device="cpu")
A = list(old.actor_mean)                       # Linear, Tanh, Linear, Tanh, Linear, Tanh, Linear(head)
LIN = [0, 2, 4]
sp2 = old.actor_logstd.exp().flatten().pow(2)  # drift in units of A's own action std
d = torch.load(a.data, map_location="cpu")
S = d["x"][d["keep"]].float()
S = S[torch.randperm(len(S))[:a.n]]
EN = [float(e) for e in a.energy.split(",")]


def aug(x):
    return torch.cat([x, torch.ones(len(x), 1)], 1)


def downstream(i):
    """Pre-activation of Linear i -> A's action, through A's own later layers and head."""
    mods = A[i + 1:]

    def f(z):
        h = z
        for m in mods:
            h = m(h)
        return h
    return f


X, Z = {}, {}                                  # inputs and pre-activations of each hidden Linear on A's states
with torch.no_grad():
    h = S
    for i, m in enumerate(A):
        if i in LIN:
            X[i] = h
            Z[i] = m(h)
        h = m(h)
    mu_old = h

C, Q, J = {}, {}, {}
for i in LIN:
    xa = aug(X[i])
    C[i] = xa.T @ xa / len(xa)
    with torch.enable_grad():
        Ji = torch.cat([vmap(jacrev(downstream(i)))(Z[i][k:k + 2000]).detach() for k in range(0, len(S), 2000)])
    Ji = Ji / sp2.sqrt()[None, :, None]        # (n, 8, 256), action rows in units of A's std
    J[i] = Ji
    Q[i] = torch.einsum("nao,nap->op", Ji, Ji) / len(Ji)
torch.set_grad_enabled(False)


def top(M, e):
    ev, V = torch.linalg.eigh(M)
    ev, V = ev.flip(0).clamp_min(0), V.flip(1)
    k = int((ev.cumsum(0) / ev.sum() < e).sum()) + 1
    return V[:, :k] @ V[:, :k].T, k


P = {e: {i: (top(C[i], e), top(Q[i], e)) for i in LIN} for e in EN}


def mixed_mu(Ws):
    """A's action through hidden layers with weights Ws[i] = (W, b), A's own old head."""
    h = S
    for i, m in enumerate(A[:-1]):
        h = h @ Ws[i][0].T + Ws[i][1] if i in LIN else m(h)
    return A[-1](h)


def drift(mu):
    return (((mu - mu_old) ** 2) / sp2).sum(1).mean().item()


res = {"n_states": len(S), "dims": {i: {"in": C[i].shape[0], "out": Q[i].shape[0]} for i in LIN}, "runs": {}}
print(f"A's states: {len(S)} | subspace sizes (input k / output k) per hidden layer at energy {EN}:")
for e in EN:
    print(f"  {e}: " + " ".join(f"L{n + 1} {P[e][i][0][1]}/{C[i].shape[0]} in, {P[e][i][1][1]}/{Q[i].shape[0]} out"
                                for n, i in enumerate(LIN)))
for c in a.ckpts.split(","):
    new = load_policy(c, 42, device="cpu")
    Nm = list(new.actor_mean)
    dW = {i: torch.cat([Nm[i].weight - A[i].weight, (Nm[i].bias - A[i].bias)[:, None]], 1) for i in LIN}
    Ws_new = {i: (Nm[i].weight, Nm[i].bias) for i in LIN}
    D_true = drift(mixed_mu(Ws_new))
    dmu = sum(torch.einsum("nao,no->na", J[i], aug(X[i]) @ dW[i].T) for i in LIN)
    D_1st = dmu.pow(2).sum(1).mean().item()
    D_fac = sum(torch.trace(dW[i] @ C[i] @ dW[i].T @ Q[i]).item() for i in LIN)
    r = {"D_true": D_true, "D_1st": D_1st, "D_fac": D_fac, "by_energy": {}}
    for e in EN:
        rows, Wo, Wg = [], {}, {}
        for i in LIN:
            (Pin, _), (Pout, _) = P[e][i]
            blk = Pout @ dW[i] @ Pin
            nrm = dW[i].pow(2).sum()
            rows.append({"rho": (blk.pow(2).sum() / nrm).item(), "gpm": ((dW[i] @ Pin).pow(2).sum() / nrm).item()})
            for tgt, upd in ((Wo, dW[i] - blk), (Wg, dW[i] - dW[i] @ Pin)):
                tgt[i] = (A[i].weight + upd[:, :-1], A[i].bias + upd[:, -1])
        r["by_energy"][e] = {"layers": rows, "D_true_ours": drift(mixed_mu(Wo)), "D_true_gpm": drift(mixed_mu(Wg))}
    res["runs"][c] = r
    name = "/".join(c.split("/")[-3:-1])
    print(f"\n{name}: A drift (old-std units) true {D_true:.4f} | first-order {D_1st:.4f} | factored {D_fac:.4f}")
    for e in EN:
        b = r["by_energy"][e]
        rho = " ".join("%.3f" % x["rho"] for x in b["layers"])
        gpm = " ".join("%.3f" % x["gpm"] for x in b["layers"])
        print(f"  energy {e}: rho per layer {rho} | GPM would block {gpm} | drift after removing the block: "
              f"ours {b['D_true_ours']:.4f}, GPM {b['D_true_gpm']:.4f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1, default=str)
