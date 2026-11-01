"""Pre-registered mechanism tests: why the projected adapter (nsa64) learns PlaceCube faster than the prefix sum with
the same initial network (nsa64npr), 0.61M vs 0.92M median t90 (projection_test.py, 11-01 11:30). Written 2026-11-01
before any run of this queue (scripts/run_why.sh) existed; the thresholds below are the registration.

Common definitions. Seeds 1-8, paired. t90 as projection_test.py: global step of the first of 3 consecutive evals with
success_once >= 0.9; a finished run (final_ckpt.pt) without one counts as the 8M halt. I = 102,400 (eval interval),
B = 51,200 (steps per iteration), d = paired difference in intervals. "X slower than Y": median d >= 1 and one-sided
exact signed-rank p <= 0.05 (projection_test.signed_rank_p: zeros dropped, average ranks). Stage counts come from
train.csv (st_*, wrwl/phase_log.py). For a run, p(K) = share of the training episodes ended in iterations 1..K that
reached the release: held the cube at the goal, or succeeded another way (dropped onto the goal), i.e.
(st_held + st_succ - st_held_succ) / st_done; K_own = t90 / B (the iterations completed before the t90 eval). Every rule
needs all 8 seeds finished and one code_md5 over the runs it reads.

Rule 0 (existing runs, no GPU; --np_runs = runs/protect_np/runs). Theorem 3 (gating) says the slowdown is p_nsa / p_npr
with p the probability of reaching the release. Every successful episode reached it, and p_nsa <= 1, so the slowdown
gating can explain is at most 1 / ts_npr, ts_npr = npr's training success over its own window (train_success weighted
by train_episodes). G_max = median log(1 / ts_npr) / median log(t90_npr / t90_nsa64). G_max <= 0.3: existing data
already exclude gating as the main mechanism; >= 0.7: not excluded; else gating can explain at most G_max.
StackCube (seeds 1-3) is printed with the same bound and does not decide. Also printed, not deciding (Rule 0 and A5):
the median ratio npr / nsa64 of the mean per-decision KL between successive policies over iterations 1-12
(train.csv kl_per_decision; --target_kl is off, so nothing caps it): a ratio well above 1 says the prefix sum takes
larger policy steps per iteration, which favours a step-size account (E1 closed by tuning).
A0 replication (nsa64, npr rerun with stage logging): each arm's t90 within one interval of protect_np's for >= 6 of 8
seeds, and npr slower than nsa64. Otherwise stop: nothing below is read.
A1 stage-1 loss: p over the common window K = 12 (614,400 steps); npr loses stage-1 competence if median
p_npr / p_nsa64 <= 0.9 and signed-rank p <= 0.05 on p_nsa64 - p_npr > 0.
A2 gating share: G = median log(p_nsa64(K_own) / p_npr(K_own)) / median log(t90_npr / t90_nsa64). G >= 0.7 and A1:
gating is the main mechanism; G <= 0.3: it is not; else partial.
A3 (registered, secondary) eval failure mix: at each seed's nsa64 t90, npr's failures 1 - success_once split into
no_release (held at the goal, still grasped at the end), lost (grasped, never held at the goal) and no_grasp. Stage-1
failures dominate if median (lost + no_grasp) / failures >= 0.6, release failures if median no_release / failures >= 0.6.
B random projection (rp, --nsa_proj random): B1 rp slower than nsa64 (the heads' rows matter); B2 rp ~ npr: |median d|
<= 1 and neither one-sided p <= 0.05.
C restoring force (rf01 / rf1 / rf10 = npr + --nsa_restore 0.1 / 1 / 10; alpha 0.05 / 3 per coefficient). Gap = median
d(npr - nsa64) of the A runs. lambda recovers the speed: median improvement u = d(npr - rf) >= gap / 2 with p <= 0.0167
and median d(rf - nsa64) <= 1. None helps: every lambda has median u < 1 or p > 0.0167. Else partial.
D stage-1 cost (nsa64d / nprd: env kwarg stage1_cost 0.05). r = log(t90_npr / t90_nsa64) per seed, sparse (A runs) and
dense; shrink = median r_dense / median r_sparse. Closes the gap (gating's prediction): shrink <= 0.5 and signed-rank
p <= 0.05 on r_sparse - r_dense > 0. No effect: shrink >= 0.8. Else partial.
E adapter learning rate (lo 1e-4, 3e-4 = A runs, hi 1e-3; head, log-std, critic stay at 3e-4). Best rate per arm =
lowest median t90 (ties: nearer 3e-4, then lower). If an arm's best is at an edge of its tested rates, the next rate out
(x1/3 or x3: nprvlo 3e-5, nprvhi 3e-3, nsa64vlo 3e-5, nsa64vhi 3e-3) is run on the same seeds before E1 is read.
E1: d* = npr at its best minus nsa64 at its best. Survives tuning: median d* >= 1 and p <= 0.05; closed by tuning:
median d* <= 0; else reduced. E2: G (as A2) at every rate where the median t90 ratio npr / nsa64 is >= 1.15: all >= 0.7
gating tracks the rate; all <= 0.3 not gating; else mixed.
F output mask (registered 11-01 15:25 in docs/conflict-reuse-2026-10-31.md, before any of these runs): the same adapter,
draw and seeds, the write multiplied by P = a N + b Pi: nsa64 (1,0), npr (1,1), vis (0,1, --nsa_mask visible), none
(0,0, --nsa_mask zero); plus nsa64z (nsa64 with a zero last adapter layer). In intervals per seed,
I = t(npr) - t(nsa64) - t(vis) + t(none). F1: median I >= 2 and one-sided p <= 0.05: not additive; |median I| <= 1:
additive; else undetermined. F2: |median d(vis - npr)| <= 1: the visible channel alone gives the prefix sum's speed
(routing); vis slower than none: the visible channel alone is harmful (close to additive); else position only.
F3: |median d(nsa64z - nsa64)| <= 1: with the projection the initialisation does not matter; nsa64z slower than nsa64:
the random initialisation contributes; else position only.
  python scripts/why_test.py --runs=runs/why/runs [--np_runs=runs/protect_np/runs]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

I, B, HALT = 102_400, 51_200, 8_000_000
K_COMMON = 12
# protect_np t90 per seed 1-8 (projection_test.py, 11-01 11:30; docs/conflict-reuse-2026-10-31.md top table)
REF = {"nsa64": [614400, 716800, 614400, 614400, 614400, 614400, 716800, 614400],
       "npr": [1024000, 921600, 921600, 921600, 819200, 819200, 1126400, 921600]}
SEEDS = range(1, 9)
# arm -> (nsa_project, nsa_init, nsa_proj, nsa_restore, adapter_lr, stage1_cost, nsa_mask)
ARMS = {"nsa64": (True, None, "heads", 0.0, None, 0.0), "npr": (False, "rand", "heads", 0.0, None, 0.0),
        "rp": (True, None, "random", 0.0, None, 0.0),
        "rf01": (False, "rand", "heads", 0.1, None, 0.0), "rf1": (False, "rand", "heads", 1.0, None, 0.0),
        "rf10": (False, "rand", "heads", 10.0, None, 0.0),
        "nsa64d": (True, None, "heads", 0.0, None, 0.05), "nprd": (False, "rand", "heads", 0.0, None, 0.05)}
ARMS = {k: v + ("null",) for k, v in ARMS.items()}
ARMS.update({"vis": (True, None, "heads", 0.0, None, 0.0, "visible"), "none": (True, None, "heads", 0.0, None, 0.0, "zero"),
             "nsa64z": (True, "zero", "heads", 0.0, None, 0.0, "null")})
LR = {"lo": 1e-4, "hi": 1e-3, "vlo": 3e-5, "vhi": 3e-3}
for base in ("nsa64", "npr"):
    for tag, lr in LR.items():
        ARMS[base + tag] = ARMS[base][:4] + (lr,) + ARMS[base][5:]
MD5 = {}


def signed_rank_p(d):
    """One-sided exact Wilcoxon signed-rank p-value for d > 0 (projection_test.py's)."""
    x = [v for v in d if v != 0]
    n = len(x)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda k: abs(x[k]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[order[j + 1]]) == abs(x[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, v in zip(ranks, x) if v > 0)
    hits = sum(sum(ranks[k] for k in range(n) if mask >> k & 1) >= w - 1e-9 for mask in range(1 << n))
    return hits / (1 << n)


def rows(path):
    raw = path.read_bytes() if path.exists() else b""
    if b"\0" in raw:
        raise SystemExit(f"{path}: NUL bytes, so two processes wrote this run; stop the duplicate, rerun it")
    lines = raw.decode().splitlines()
    return list(csv.DictReader(lines[:-1] if raw and not raw.endswith(b"\n") else lines))


def t90_of(ev, run):
    steps = [int(float(r["global_step"])) for r in ev]
    if steps != [k * I for k in range(len(steps))]:
        raise SystemExit(f"{run}/eval.csv: evals are not every {I} steps from 0")
    s = [float(r["success_once"]) for r in ev]
    for i in range(len(s) - 2):
        if all(v >= 0.9 for v in s[i:i + 3]):
            return steps[i]
    if not steps or steps[-1] < (HALT - 1) // I * I:
        raise SystemExit(f"{run}: final_ckpt.pt, no t90, and eval.csv ends before the halt's last eval")
    return HALT


def load(runs, arm, seed, check=True, task="place"):
    """{'t90', 'tr', 'ev'} of a finished run, None if unfinished; validates args.json against the arm."""
    run = runs / f"{task}_{arm}_s{seed}"
    if not (run / "final_ckpt.pt").exists():
        return None
    j = json.load(open(run / "args.json"))
    if check:
        proj, init, how, rest, alr, cost, mask = ARMS[arm]
        want = {"exp_name": run.name, "seed": seed, "env_id": "PlaceCube-v1", "halt_at": HALT, "stop_success": 0.9,
                "stop_patience": 3, "num_envs": 1024, "num_steps": 50, "eval_freq": 2, "num_eval_envs": 256,
                "learning_rate": 3e-4, "gamma": 0.95, "min_grip_std": 1.0, "nullspace_adapter": 64,
                "nsa_project": proj, "nsa_init": init, "nsa_proj": how, "nsa_restore": rest, "adapter_lr": alr,
                "head_lr": None, "nsa_freeze_adapter": False, "nsa_mask": mask}
        bad = {k: (j.get(k), v) for k, v in want.items() if j.get(k) != v}
        ek = json.loads(j.get("env_kwargs", "{}"))
        if ek.get("stage1_cost", 0.0) != cost or ek.get("reward_mode") != "sparse":
            bad["env_kwargs"] = (j.get("env_kwargs"), f"sparse, stage1_cost {cost}")
        if bad:
            raise SystemExit(f"{run}: args.json (found, expected) {bad}")
        MD5[run.name] = str(j.get("code_md5"))
    ev = rows(run / "eval.csv")
    return {"t90": t90_of(ev, run), "tr": rows(run / "train.csv"), "ev": ev}


def table(runs, arm, check=True):
    return {s: load(runs, arm, s, check) for s in SEEDS}


def ready(*tabs):
    return all(t[s] is not None for t in tabs for s in SEEDS)


def p_held(run, K):
    """p(K): the share of training episodes ended in iterations 1..K that reached the release (held at the goal, or
    succeeded without that)."""
    tr = [r for r in run["tr"] if int(r["iteration"]) <= K]
    done = sum(float(r["st_done"]) for r in tr)
    reach = sum(float(r["st_held"]) + float(r["st_succ"]) - float(r["st_held_succ"]) for r in tr)
    return reach / done if done else float("nan")


def k_own(run):
    return max(1, run["t90"] // B)


def paired(x, y, scale=I):
    """d per seed (x - y) / scale, its median and the one-sided signed-rank p for d > 0 and for d < 0."""
    d = [(x[s]["t90"] - y[s]["t90"]) / scale for s in SEEDS]
    return d, float(np.median(d)), signed_rank_p(d), signed_rank_p([-v for v in d])


def fmt(d):
    return " ".join(f"{v:5.1f}" for v in d)


def share(a, b):
    """A2 / E2: G = median log(p_a(K_own) / p_b(K_own)) / median log(t90_b / t90_a), a the projected arm."""
    num = float(np.median([math.log(p_held(a[s], k_own(a[s])) / p_held(b[s], k_own(b[s]))) for s in SEEDS]))
    den = float(np.median([math.log(b[s]["t90"] / a[s]["t90"]) for s in SEEDS]))
    return num, den, (num / den if den > 0 else float("nan"))


def kl_ratio(a, b, seeds, K=K_COMMON):
    """Median over seeds of mean kl_per_decision over iterations 1..K, b / a."""
    def m(run):
        return float(np.mean([float(r["kl_per_decision"]) for r in run["tr"] if int(r["iteration"]) <= K]))
    return float(np.median([m(b[s]) / m(a[s]) for s in seeds]))


def one_md5(names):
    v = {MD5[n] for n in names if n in MD5}
    return len(v) <= 1, sorted(v)


def rule0(np_runs, out, task="place", seeds=SEEDS):
    nsa = {s: load(np_runs, "nsa64", s, False, task) for s in seeds}
    npr = {s: load(np_runs, "nsa64npr", s, False, task) for s in seeds}
    if any(v is None for v in [*nsa.values(), *npr.values()]):
        return out(f"Rule 0 ({task}): open (protect_np runs missing or unfinished)")
    if any(not v["tr"] for v in [*nsa.values(), *npr.values()]):
        return out(f"Rule 0 ({task}): open (a protect_np run has no train.csv rows)")
    ts = []
    for s in seeds:
        tr = [r for r in npr[s]["tr"] if int(r["iteration"]) <= k_own(npr[s])]
        n = sum(float(r["train_episodes"]) for r in tr)
        ts.append(sum(float(r["train_success"]) * float(r["train_episodes"]) for r in tr) / n)
    num = float(np.median([math.log(1 / v) for v in ts]))
    den = float(np.median([math.log(npr[s]["t90"] / nsa[s]["t90"]) for s in seeds]))
    g = num / den
    out(f"Rule 0 ({task}{', reported only' if task != 'place' else ''}; protect_np, no GPU): npr training success over its own window per seed: "
        f"{' '.join(f'{v:.3f}' for v in ts)}; slowdown gating can explain <= 1/ts (median {math.exp(num):.3f}) vs "
        f"observed median t90 ratio {math.exp(den):.3f}: G_max = {g:.2f} -> "
        + ("existing data exclude gating as the main mechanism" if g <= 0.3 else
           "gating not excluded by existing data" if g >= 0.7 else f"gating can explain at most {g:.2f} of it"))
    if "kl_per_decision" in npr[seeds[0]]["tr"][0]:
        out(f"  ({task}, descriptive) per-decision KL per iteration, iterations 1-{K_COMMON}, median ratio npr / nsa64: "
            f"{kl_ratio(nsa, npr, seeds):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/why/runs")
    ap.add_argument("--np_runs", default="")
    a = ap.parse_args()
    out = print
    if a.np_runs:
        rule0(Path(a.np_runs), out)
        rule0(Path(a.np_runs), out, "stack", range(1, 4))
    runs = Path(a.runs)
    T = {arm: table(runs, arm) for arm in ARMS}
    for arm in ARMS:
        got = [T[arm][s]["t90"] / 1e6 if T[arm][s] else None for s in SEEDS]
        if any(v is not None for v in got):
            med = f"{np.median(got):.2f}" if None not in got else "open"
            out(f"  {arm:9s} t90 (M): " + " ".join("  open" if v is None else f"{v:6.2f}" for v in got)
                + f"   median {med}")
    nsa, npr = T["nsa64"], T["npr"]
    if not ready(nsa, npr):
        return out("A: open (nsa64 / npr unfinished); nothing below is read")
    ok, md5 = one_md5([f"place_{x}_s{s}" for x in ("nsa64", "npr") for s in SEEDS])
    if not ok:
        return out(f"A: open, nsa64 and npr ran under different code versions {md5}")
    rep = [sum(abs(T[x][s]["t90"] - REF[x][s - 1]) <= I for s in SEEDS) for x in ("nsa64", "npr")]
    d, med, p, _ = paired(npr, nsa)
    a0 = min(rep) >= 6 and med >= 1 and p <= 0.05
    out(f"A0 replication: seeds within one interval of protect_np: nsa64 {rep[0]}/8, npr {rep[1]}/8; "
        f"d(npr - nsa64) {fmt(d)} median {med:.1f} p {p:.4f} -> {'replicated' if a0 else 'NOT replicated: stop'}")
    if not a0:
        return
    gap = med
    pn = [p_held(nsa[s], K_COMMON) for s in SEEDS]
    pr = [p_held(npr[s], K_COMMON) for s in SEEDS]
    ratio = float(np.median([y / x for x, y in zip(pn, pr)]))
    pa1 = signed_rank_p([x - y for x, y in zip(pn, pr)])
    a1 = ratio <= 0.9 and pa1 <= 0.05
    out(f"A1 p(reached the release), iterations 1-{K_COMMON}: nsa64 {' '.join(f'{v:.3f}' for v in pn)} | npr "
        f"{' '.join(f'{v:.3f}' for v in pr)} | median ratio npr/nsa64 {ratio:.3f}, p {pa1:.4f} -> "
        + ("npr loses stage-1 competence" if a1 else "no stage-1 loss shown"))
    num, den, g = share(nsa, npr)
    out(f"A2 gating share G = {num:.3f} / {den:.3f} = {g:.2f} -> "
        + ("gating is the main mechanism" if g >= 0.7 and a1 else "gating is not the main mechanism" if g <= 0.3
           else "partial" + ("" if a1 else " (A1 not met, so not the main mechanism)")))
    mix = []
    for s in SEEDS:
        at = {int(float(r["global_step"])): r for r in npr[s]["ev"]}.get(nsa[s]["t90"])
        if at is None:
            continue
        f = 1 - float(at["success_once"])
        if f > 0:
            mix.append(((float(at["ev_lost"]) + float(at["ev_no_grasp"])) / f, float(at["ev_no_release"]) / f))
    if mix:
        s1, nr = float(np.median([m[0] for m in mix])), float(np.median([m[1] for m in mix]))
        out(f"A3 npr's eval failures at nsa64's t90 ({len(mix)} seeds with failures): stage-1 share {s1:.2f}, "
            f"no-release share {nr:.2f} -> " + ("stage-1 failures dominate" if s1 >= 0.6 else
                                                 "release failures dominate" if nr >= 0.6 else "mixed"))
    if "dev_arm_carry" in nsa[1]["tr"][0]:     # A4, descriptive: the critique's phi in stage-1 states, and the release
        def avg(run, col, K=K_COMMON):
            v = [float(r[col]) for r in run["tr"] if int(r["iteration"]) <= K and r[col] != "nan"]
            return float(np.mean(v)) if v else float("nan")
        drift = [(avg(npr[s], "dev_arm_carry") + avg(npr[s], "dev_grip_carry"))
                 / (avg(nsa[s], "dev_arm_carry") + avg(nsa[s], "dev_grip_carry")) for s in SEEDS]
        sel = {x: float(np.median([avg(T[x][s], "mu_grip_goal", 12) - avg(T[x][s], "mu_grip_carry", 12)
                                   for s in SEEDS])) for x in ("nsa64", "npr")}
        out(f"A4 (descriptive) carry-state action drift from the start policy, iterations 1-{K_COMMON}, npr / nsa64: "
            f"median {np.median(drift):.2f}; gripper mean at the goal minus in carry (release selectivity): "
            f"nsa64 {sel['nsa64']:.2f}, npr {sel['npr']:.2f}")
    if "kl_per_decision" in nsa[1]["tr"][0]:
        out(f"A5 (descriptive) per-decision KL per iteration, iterations 1-{K_COMMON}, median ratio npr / nsa64: "
            f"{kl_ratio(nsa, npr, SEEDS):.2f}")
    if ready(T["rp"]):
        d1, m1, p1, _ = paired(T["rp"], nsa)
        d2, m2, p2u, p2d = paired(T["rp"], npr)
        b1, b2 = m1 >= 1 and p1 <= 0.05, abs(m2) <= 1 and p2u > 0.05 and p2d > 0.05
        out(f"B random projection: d(rp - nsa64) {fmt(d1)} median {m1:.1f} p {p1:.4f}; d(rp - npr) {fmt(d2)} median "
            f"{m2:.1f} (p slower {p2u:.4f}, p faster {p2d:.4f}) -> "
            + ("the heads' rows matter; a same-rank random projection acts like none" if b1 and b2 else
               "the heads' rows matter; random projection in between" if b1 else
               "a random same-rank projection is not slower: the invisibility account fails (check P_r first)"))
    rf = [x for x in ("rf01", "rf1", "rf10") if ready(T[x])]
    if len(rf) == 3:
        res = []
        for x in rf:
            du, mu_, pu, _ = paired(npr, T[x])
            dd, md, _, _ = paired(T[x], nsa)
            ps = float(np.median([p_held(T[x][s], k_own(T[x][s])) for s in SEEDS]))
            res.append((x, mu_, pu, md, ps))
            out(f"C {x}: improvement over npr {fmt(du)} median {mu_:.1f} p {pu:.4f}; d({x} - nsa64) median {md:.1f}; "
                f"median p(held) own window {ps:.3f}")
        rec = [x for x, mu_, pu, md, _ in res if mu_ >= gap / 2 and pu <= 0.05 / 3 and md <= 1]
        none = all(mu_ < 1 or pu > 0.05 / 3 for _, mu_, pu, _, _ in res)
        out("C verdict: " + (f"a data-free restoring force recovers the speed ({', '.join(rec)})" if rec else
                             "no coefficient helps" if none else "partial (helps, not to within one interval)"))
    if ready(T["nsa64d"], T["nprd"]):
        rs = [math.log(npr[s]["t90"] / nsa[s]["t90"]) for s in SEEDS]
        rd = [math.log(T["nprd"][s]["t90"] / T["nsa64d"][s]["t90"]) for s in SEEDS]
        shrink = float(np.median(rd)) / float(np.median(rs)) if np.median(rs) > 0 else float("nan")
        pd_ = signed_rank_p([x - y for x, y in zip(rs, rd)])
        out(f"D stage-1 cost: log ratio sparse {' '.join(f'{v:.2f}' for v in rs)} | dense "
            f"{' '.join(f'{v:.2f}' for v in rd)}; shrink {shrink:.2f}, p {pd_:.4f} -> "
            + ("closes the gap (gating's prediction holds)" if shrink <= 0.5 and pd_ <= 0.05 else
               "no effect" if shrink >= 0.8 else "partial"))
        for x, y in (("nsa64d", "nsa64"), ("nprd", "npr")):
            dd, md, _, _ = paired(T[x], T[y])
            out(f"  d({x} - {y}) {fmt(dd)} median {md:.1f}")
    rates = {"lo": 1e-4, "": 3e-4, "hi": 1e-3, "vlo": 3e-5, "vhi": 3e-3}
    lr_ready = all(ready(T[f"{b}{t}"]) for b in ("nsa64", "npr") for t in ("lo", "hi"))
    if lr_ready:
        best, ext = {}, []
        for b in ("nsa64", "npr"):
            tested = [t for t in ("vlo", "lo", "", "hi", "vhi") if ready(T[b + t])]
            meds = {t: float(np.median([T[b + t][s]["t90"] for s in SEEDS])) for t in tested}
            t_best = min(tested, key=lambda t: (meds[t], abs(math.log(rates[t] / 3e-4)), rates[t]))
            best[b] = t_best
            out(f"E {b}: median t90 by adapter lr " + ", ".join(f"{rates[t]:.0e}: {meds[t] / 1e6:.2f}M" for t in tested)
                + f" -> best {rates[t_best]:.0e}")
            if t_best == tested[0] and t_best != "vlo":
                ext.append(b + ("vlo" if t_best == "lo" else "lo"))
            if t_best == tested[-1] and t_best != "vhi":
                ext.append(b + ("vhi" if t_best == "hi" else "hi"))
        if ext:
            out(f"E1: open, the best rate is at an edge: run {' '.join(ext)} on seeds 1-8 (registered extension)")
        else:
            d_, m_, p_, _ = paired(T["npr" + best["npr"]], T["nsa64" + best["nsa64"]])
            out(f"E1 best vs best: d* {fmt(d_)} median {m_:.1f} p {p_:.4f} -> "
                + ("the gap survives adapter-lr tuning" if m_ >= 1 and p_ <= 0.05 else
                   "closed by tuning" if m_ <= 0 else "reduced, not closed"))
        gs = []
        for t in ("lo", "", "hi"):
            a_, b_ = T["nsa64" + t], T["npr" + t]
            ratio_t = float(np.median([b_[s]["t90"] / a_[s]["t90"] for s in SEEDS]))
            num, den, g = share(a_, b_)
            out(f"E2 lr {rates[t]:.0e}: median t90 ratio npr/nsa64 {ratio_t:.2f}, G {g:.2f}")
            if ratio_t >= 1.15:
                gs.append(g)
        if gs:
            out("E2 verdict: " + ("gating tracks the rate" if all(v >= 0.7 for v in gs) else
                                  "not gating at any rate" if all(v <= 0.3 for v in gs) else "mixed"))
    if ready(T["vis"], T["none"]):
        inter = [(npr[s]["t90"] - nsa[s]["t90"] - T["vis"][s]["t90"] + T["none"][s]["t90"]) / I for s in SEEDS]
        mi, pi = float(np.median(inter)), signed_rank_p(inter)
        out(f"F1 interaction I = t(npr) - t(nsa64) - t(vis) + t(none): {fmt(inter)} median {mi:.1f} p {pi:.4f} -> "
            + ("not additive" if mi >= 2 and pi <= 0.05 else "additive" if abs(mi) <= 1 else "undetermined"))
        d2, m2, _, _ = paired(T["vis"], npr)
        d3, m3, p3, _ = paired(T["vis"], T["none"])
        out(f"F2 d(vis - npr) {fmt(d2)} median {m2:.1f}; d(vis - none) {fmt(d3)} median {m3:.1f} p {p3:.4f} -> "
            + ("the visible channel alone gives the prefix sum's speed (routing)" if abs(m2) <= 1 else
               "the visible channel alone is harmful (close to additive)" if m3 >= 1 and p3 <= 0.05 else
               "position only"))
    if ready(T["nsa64z"]):
        d4, m4, p4, _ = paired(T["nsa64z"], nsa)
        out(f"F3 d(nsa64z - nsa64) {fmt(d4)} median {m4:.1f} p {p4:.4f} -> "
            + ("with the projection the initialisation does not matter" if abs(m4) <= 1 else
               "the random initialisation contributes" if m4 >= 1 and p4 <= 0.05 else "position only"))
    ok, md5 = one_md5(list(MD5))
    if not ok:
        out(f"WARNING: the runs read span code versions {md5}; verdicts that mix them are not registered")


if __name__ == "__main__":
    main()
