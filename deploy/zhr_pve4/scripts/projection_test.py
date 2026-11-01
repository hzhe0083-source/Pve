"""The pre-registered projection test (docs/conflict-reuse-2026-10-31.md, section of 11-01 10:45), written before any
result was looked at. For each seed s on PlaceCube (seeds 1-8, all arms in one directory under one code version,
which args.json must show, with this queue's flags):
    d_s = (t90 of a no-projection arm - t90 of nsa64) / 102400        (in eval intervals)
t90 = global step of the first of 3 consecutive evals with success_once >= 0.9 (as analyze_protect.py and
ret_watch.sh); a run that ended (final_ckpt.pt) without one counts as the halt, 8M. The projection is credited with
faster learning only if BOTH nsa64np and nsa64npr have median(d) >= 1 and a one-sided exact Wilcoxon signed-rank
p <= 0.05 (zeros dropped, average ranks for ties); otherwise, for speed, it is an implementation detail (the default).
A run without final_ckpt.pt is unfinished; unfinished seeds leave the verdict open unless the other no-projection
arm has already failed. (Amended 11-01 10:44, found by testing this script on synthetic runs; one
no-projection run, place_nsa64np_s1, had finished by then and its result was not looked at. The first version used a
one-sided 95% t lower bound of mean(d), which one censored seed at 8M (d ~ 71) turns negative, so a prefix sum that
fails outright would have counted against the projection; the t bound is still printed.)
Stack (seeds 1-3) is printed but does not decide. The frozen-adapter rule (nsa64fz, its own directory) is printed when
--fz is given: median t90 within 0.1M (100,000 steps as registered, less than one eval interval) of nsa64's median
-> random null-space features; within 0.1M of head-only (1,228,800, runs/protect_c on haha-c) -> the trained adapter;
otherwise both are reported, labelled in between or outside both windows (rule 3 registers only in between). A second
line gives the unregistered reading of 0.1M as one eval interval.
  python scripts/projection_test.py --runs=runs/protect_np/runs [--fz=runs/protect_fz/runs]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

INTERVAL = 102_400
HALT = 8_000_000
HEAD_ONLY = 1_228_800       # place_head median t90 (runs/protect_c, haha-c, 11-01 06:20)
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753}    # one-sided 95% t quantiles by degrees of freedom


ENV = {"place": "PlaceCube-v1", "stack": "StackCubeNarrow-v1"}
MD5 = {}        # run name -> code_md5 (args.json) of every finished run read


def t90(run, halt=HALT):
    """(t90 or None, finished). Only a run with final_ckpt.pt is finished: one without is running, or was stopped and
    will be rerun from scratch. A finished run without 3 consecutive evals >= 0.9 counts as the halt, which it must
    have reached under this queue's flags; anything else stops the script with the run named."""
    if not (run / "final_ckpt.pt").exists():
        return None, False
    task, arm, seed = run.name.split("_")
    j = json.load(open(run / "args.json"))
    want = {"exp_name": run.name, "seed": int(seed[1:]), "env_id": ENV[task], "halt_at": halt, "stop_success": 0.9,
            "stop_patience": 3, "stop_after": 0, "num_envs": 1024, "num_steps": 50, "eval_freq": 2,
            "num_eval_envs": 256, "nullspace_adapter": 64, "nsa_project": arm in ("nsa64", "nsa64fz"),
            "nsa_init": "rand" if arm == "nsa64npr" else None, "nsa_freeze_adapter": arm == "nsa64fz"}
    bad = {k: (j.get(k), v) for k, v in want.items() if j.get(k) != v}
    if bad:
        raise SystemExit(f"{run}: args.json (found, expected) {bad}")
    MD5[run.name] = str(j.get("code_md5"))
    raw = (run / "eval.csv").read_bytes() if (run / "eval.csv").exists() else b""
    if b"\0" in raw:    # two processes wrote this run: csv stops on NUL in Python 3.10 and skips rows in 3.11+
        raise SystemExit(f"{run}/eval.csv: NUL bytes, so two processes wrote this run; stop the duplicate, rerun it")
    lines = raw.decode().splitlines()
    rows = list(csv.DictReader(lines[:-1] if raw and not raw.endswith(b"\n") else lines))  # drop a row being written
    steps = [int(float(r["global_step"])) for r in rows]
    if steps != [k * INTERVAL for k in range(len(steps))]:
        raise SystemExit(f"{run}/eval.csv: evals are not every {INTERVAL} steps from 0: {steps}")
    s = [float(r["success_once"]) for r in rows]
    for i in range(len(s) - 2):
        if all(v >= 0.9 for v in s[i:i + 3]):      # nan fails, as in ppo.py's early stop
            return steps[i], True
    last = (halt - 1) // INTERVAL * INTERVAL        # ppo.py's last eval before --halt_at: 7,987,200 for 8M
    if not steps or steps[-1] < last:
        raise SystemExit(f"{run}: final_ckpt.pt, no t90, and eval.csv ends at {steps[-1] if steps else 'nothing'}, "
                         f"not at the halt's last eval {last}")
    return halt, True


def table(runs, task, arms, seeds, halt=HALT):
    out = {}
    for a in arms:
        for s in seeds:
            r = runs / f"{task}_{a}_s{s}"
            out[a, s] = t90(r, halt) if r.exists() else (None, False)
    return out


def signed_rank_p(d):
    """One-sided exact Wilcoxon signed-rank p-value for d > 0: zeros dropped, average ranks for ties, every sign flip."""
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


def paired(T, arm, seeds):
    """d per seed (None if either run is unfinished), its median, mean, one-sided 95% t lower bound, signed-rank p."""
    d = []
    for s in seeds:
        (x, _), (y, _) = T[arm, s], T["nsa64", s]
        d.append(None if x is None or y is None else (x - y) / INTERVAL)
    got = [v for v in d if v is not None]
    if len(got) < len(seeds) or len(got) < 2:
        return d, None, None, None, None
    m, sd = float(np.mean(got)), float(np.std(got, ddof=1))
    lb = m - T95[len(got) - 1] * sd / math.sqrt(len(got)) if sd > 0 else m
    return d, float(np.median(got)), m, lb, signed_rank_p(got)


def fmt(v):
    return "  open" if v is None else f"{v / 1e6:6.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/protect_np/runs")
    p.add_argument("--fz", default="")
    a = p.parse_args()
    runs = Path(a.runs)
    if not list(runs.glob("place_*_s*")):
        raise SystemExit(f"no place_*_s* run directories in {runs}: pass the queue's runs/ directory")
    arms = ["nsa64", "nsa64np", "nsa64npr"]
    for task, seeds, decides in (("place", range(1, 9), True), ("stack", range(1, 4), False)):
        T = table(runs, task, arms, seeds)
        print(f"\n{task} ({'decides' if decides else 'reported only'}): t90 in M steps per seed "
              f"(8.00 = ended without reaching 0.9; open = not finished)")
        for arm in arms:
            vals = [T[arm, s][0] for s in seeds]
            fin = [v for v in vals if v is not None]
            med = f"{np.median(fin) / 1e6:.2f}" if len(fin) == len(vals) else "open"
            print(f"  {arm:9s} " + " ".join(fmt(v) for v in vals) + f"   median {med}")
        verdicts = []
        for arm in ("nsa64np", "nsa64npr"):
            d, med, m, lb, pw = paired(T, arm, seeds)
            ds = " ".join("  open" if v is None else f"{v:6.1f}" for v in d)
            if med is None:
                print(f"  d({arm} - nsa64) in intervals: {ds}   -> open (unfinished seeds)")
                verdicts.append(None)
                continue
            yes = med >= 1 and pw <= 0.05
            label = ("slower without projection" if yes else "not shown") if decides else "no test (3 seeds: p >= 1/8)"
            print(f"  d({arm} - nsa64) in intervals: {ds}   median {med:.1f}, signed-rank p {pw:.4f} "
                  f"(mean {m:.2f}, t lower bound {lb:.2f}) -> {label}")
            verdicts.append(yes)
        if decides:
            md5 = {MD5[n] for n in MD5 if n.startswith(task + "_")}
            if len(md5) > 1:
                print(f"  VERDICT: open (runs from different code versions: {sorted(md5)})")
            elif False in verdicts:     # one arm has already failed, so 'both slower' cannot hold
                print("  VERDICT: for speed, the projection is an implementation detail (pre-registered default)"
                      + ("; the other no-projection arm is still open" if None in verdicts else ""))
            elif None in verdicts:
                print("  VERDICT: open (some seeds unfinished)")
            else:
                print("  VERDICT: the projection makes learning faster (both no-projection arms slower, pre-registered rule)")
    if a.fz:
        if not Path(a.fz).is_dir() or (Path(a.fz) / "runs").is_dir():
            raise SystemExit(f"--fz={a.fz}: pass the fz queue's runs/ directory (it exists once that queue starts)")
        fz = table(Path(a.fz), "place", ["nsa64fz"], range(1, 4), halt=3_000_000)
        ref = table(runs, "place", ["nsa64"], range(1, 9))
        vf = [fz["nsa64fz", s][0] for s in range(1, 4)]
        vr = [ref["nsa64", s][0] for s in range(1, 9)]
        print("\nfrozen adapter (place, halt 3M): " + " ".join(fmt(v) for v in vf))
        if None in vf or None in vr:
            print("  VERDICT: open (unfinished runs)")
        else:
            mf, mr = float(np.median(vf)), float(np.median(vr))
            lo, hi = min(mr, HEAD_ONLY), max(mr, HEAD_ONLY)

            def rule3(w):       # registered: w = 0.1M = 100,000 steps, less than one eval interval
                near_nsa, near_head = abs(mf - mr) <= w, abs(mf - HEAD_ONLY) <= w
                return ("speed comes from random null-space features" if near_nsa and not near_head else
                        "speed comes from the trained adapter" if near_head and not near_nsa else
                        "report both (near both references)" if near_nsa else
                        "report both (in between)" if lo < mf < hi else
                        "report both (outside both windows: faster than both references)" if mf < lo else
                        "report both (outside both windows: slower than both references"
                        + (", median censored at the 3M halt)" if mf >= 3_000_000 else ")"))
            md5 = {MD5[n] for n in MD5 if n.startswith(("place_nsa64fz_", "place_nsa64_"))}
            if len(md5) > 1:
                print(f"  note: nsa64fz and nsa64 ran under different code versions {sorted(md5)}")
            dn, dh = abs(mf - mr) / INTERVAL, abs(mf - HEAD_ONLY) / INTERVAL
            print(f"  median {mf / 1e6:.2f}M vs nsa64 {mr / 1e6:.2f}M ({dn:g} intervals) and head-only "
                  f"{HEAD_ONLY / 1e6:.2f}M ({dh:g} intervals) -> {rule3(100_000)}")
            print(f"  not registered: '0.1M' read as one eval interval (102,400, inclusive) -> {rule3(INTERVAL)}")


if __name__ == "__main__":
    main()
