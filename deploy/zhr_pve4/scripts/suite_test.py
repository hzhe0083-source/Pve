"""The pre-registered analysis of the task suite (written 2026-11-01, before any suite run, gate or pilot existed).
Which tasks: scripts/suite_gate.py (zero-shot <= 0.30, pilot t90 <= 4M), frozen in suite.json before the first
confirmatory run, in three strata: counted (a new object or contact mode: rules 3-4), near (the 4 cm cube under a
development task's success test: reported), dev (place, stack: the method and these rules were designed on them;
reported, R0). Every number below is computed by this script; nothing is chosen after the data are seen.

0 Data. Single-step runs <task>_<arm>_s<seed> (scripts/run_protect.sh, PROT_DIR=runs/suite), seeds 1-8 paired across
  arms (same seed = same env starts and torch seed), arms nsa64 (null-space adapter, width 64), nsa64npr (prefix sum
  from nsa64's own random adapter), monitor (full fine-tuning), head (output layer only); optional fncl (FN-PPO).
  Every run must carry the queue's flags in args.json (env id, kwargs and adapter as suite.json froze them), and all
  runs one code_md5; otherwise no verdict.
1 t90 = global step of the first of 3 consecutive evals with success_once >= 0.9 (evals every 102,400 steps from 0;
  256 envs; deterministic). A finished run (final_ckpt.pt) without one is censored at the halt, 8,000,000; a run
  without final_ckpt.pt is open.
2 Per task, comparator c in {nsa64npr, monitor, head}: d_s = (t90_c - t90_nsa64) / 102,400. "nsa64 faster than c" iff
  median(d) >= 1 and the one-sided exact Wilcoxon signed-rank p <= 0.05 (zeros dropped, average ranks for ties);
  "c faster than nsa64" iff median(d) <= -1 and the mirrored p <= 0.05; else "no difference shown".
3 Over the K counted tasks, per comparator c:
  3a (deciding) "nsa64 faster than c across the suite" iff K >= 3, the stratified exact signed-rank test (sum over
     tasks of each task's W+, exact null by convolving the per-task sign-flip distributions) gives p <= 0.05, and no
     counted task has "c faster than nsa64" (rule 2). K <= 2: not testable. Open while any counted run is open.
  3b (decides only the wording "generalises to new tasks") one-sided exact sign test over tasks: n+ = tasks with
     median(d) >= 1, p = P(Binomial(K, 1/2) >= n+) <= 0.05; K = 5 needs 5/5; K <= 4 cannot pass (untestable).
     Open tasks: decided only if the verdict is the same with every open seed at -inf and at +inf.
4 Headline: "nsa64 is faster across the suite than both the prefix sum and full fine-tuning" iff 3a holds for
  c = nsa64npr AND c = monitor; "... than <c> only" if exactly one holds; "not shown" if neither; with 3b for both,
  "and the speed-up generalises to new tasks". c = head is reported (the new-capacity check). Reported, not deciding:
  rule 2 on every counted task (intersection-union), per-task median-t90 ratios c / nsa64 and their geometric mean,
  the same tests over counted + near tasks.
5 Exact retention, every nsa64 run of every stratum: (a) ppo.py's end-of-run gap (old head on the adapted vs the base
  features, Pick's closed-loop states) <= 1e-4 and (b) retention_eval.py old_head on final_ckpt.pt >= the expert's
  success - 2/256. "Exact retention holds across the suite" iff every run passes both. A failure is a bug: no retention
  claim until it is found.
6 Forgetting under fine-tuning: per task, "full fine-tuning forgets" iff the median over seeds of monitor's old_head
  on final_ckpt.pt is < 0.95.
7 Chain (scripts/run_chain.sh, suite.json's chain; step 1 = the single-step run; runs <arm>-<k>-<task>_s<seed>; a chain
  stops at the first step without 3 evals >= 0.9). 7a (decides the chain's retention claim), arms chain64 and
  chain64p: every LEARNED skill (its own network's success >= 0.9 on nsa_eval.py's starts) keeps, for every seed,
  success through the chain's last level >= its own network's - 2/256 and largest action change <= 1e-4. Reported with
  it: learned skills per chain, chains stopped and where. Reported: 7b per step k >= 2, d = (t90 single-step nsa64 -
  t90 chain64p step k) / 102,400 (both heads start as Pick's; > 0 = reading the earlier adapters helps), median and
  signed-rank p; 7c t90 against the free width 256 - 8k at level k, chain64p only (chain64 inherits each previous
  head and log-std, so its t90 mixes capacity with the start); 7d chain64npr (own level and last level) and ftchain.
8 Parameters added per skill (actor): nsa64 37,832 (26.2% of Pick's 144,648), head 2,056, monitor 0, npr as nsa64.
R0 Regression (checked before anything else is read): place_nsa64 seeds 1-8 under the suite's code reproduce
  runs/protect_np's t90 (6 7 6 6 6 6 7 6 intervals) within one interval each; otherwise stop and find out why.
C Cut-off (--cutoff, used only if the queue has not finished by 2026-11-25 23:59, decided before any confirmatory
  result is read): the analysis keeps seeds 1..m for every task and arm, m the largest number such that every
  single-step run of the suite (all strata, the four arms) with seed <= m is finished; the chain keeps seeds <= m too.
  m <= 4: rule 2 cannot reach p <= 0.05 (smallest p = 1/2^m), reported as such.
  python scripts/suite_test.py --runs=runs/suite/runs --suite=runs/suite/suite.json --ret=runs/suite/ret \
      --logs=runs/suite [--dev] [--chain=runs/suite_chain] [--cutoff] [--fncl] [--out=x.json]
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

INTERVAL, HALT = 102_400, 8_000_000
ARMS, COMPARATORS = ["nsa64", "nsa64npr", "monitor", "head"], ["nsa64npr", "monitor", "head"]
TOL_GAP, TOL_EP, LEARNED = 1e-4, 2 / 256, 0.9
PLACE_PROTECT_NP = [6, 7, 6, 6, 6, 6, 7, 6]     # place_nsa64 t90 / 102,400, seeds 1-8 (runs/protect_np, 11-01 11:30)
GAP_RE = re.compile(r"nullspace adapter: max \|old head on h~ - old head on h\| = ([0-9.eE+-]+)")
CHAIN_ARMS = ["chain64", "chain64p", "chain64npr", "ftchain"]
FIRST_ARM = {"chain64": "nsa64", "chain64p": "nsa64", "chain64npr": "nsa64npr", "ftchain": "monitor"}
MD5 = {}
SEEDS = list(range(1, 9))


def want_args(task, arm, seed, suite):
    w = {"seed": seed, "env_id": suite["env"][task], "obs_adapter": suite["obs_adapter"][task], "halt_at": HALT,
         "stop_success": 0.9, "stop_patience": 3, "stop_after": 0, "num_envs": 1024, "num_steps": 50, "eval_freq": 2,
         "num_eval_envs": 256, "gamma": 0.95, "min_grip_std": 1.0, "target_kl": 1e9, "bootstrap_at_done": "truncated",
         "init_actor_only": True, "learning_rate": 3e-4}
    if arm in ("nsa64", "nsa64npr"):
        w.update(nullspace_adapter=64, nsa_project=arm == "nsa64", nsa_init="rand" if arm == "nsa64npr" else None,
                 protect_old="none", freeze_trunk=False)
    elif arm == "monitor":
        w.update(nullspace_adapter=0, protect_old="monitor", freeze_trunk=False)
    elif arm == "head":
        w.update(nullspace_adapter=0, protect_old="none", freeze_trunk=True)
    elif arm == "fncl":
        w.update(nullspace_adapter=0, protect_old="fn", freeze_trunk=False)
    return w


def read_eval(run):
    raw = (run / "eval.csv").read_bytes() if (run / "eval.csv").exists() else b""
    if b"\0" in raw:
        raise SystemExit(f"{run}/eval.csv: NUL bytes, so two processes wrote this run; stop the duplicate, rerun it")
    lines = raw.decode().splitlines()
    return list(csv.DictReader(lines[:-1] if raw and not raw.endswith(b"\n") else lines))


def t90(run, want=None, kwargs=None):
    """t90 (censored at HALT), or None if the run is open. With want / kwargs, args.json must match them."""
    if not (run / "final_ckpt.pt").exists():
        return None
    j = json.load(open(run / "args.json"))
    if want is not None:
        bad = {k: (j.get(k), v) for k, v in want.items()
               if not (j.get(k) == v or (isinstance(v, float) and isinstance(j.get(k), (int, float))
                                         and math.isclose(j.get(k), v, rel_tol=1e-9)))}
        if kwargs is not None and json.loads(j.get("env_kwargs") or "{}") != kwargs:
            bad["env_kwargs"] = (j.get("env_kwargs"), kwargs)
        if bad:
            raise SystemExit(f"{run}: args.json (found, expected) {bad}")
    MD5[run.name] = str(j.get("code_md5"))
    rows = read_eval(run)
    steps = [int(float(r["global_step"])) for r in rows]
    if steps != [k * INTERVAL for k in range(len(steps))]:
        raise SystemExit(f"{run}/eval.csv: evals are not every {INTERVAL} steps from 0: {steps[:5]}...")
    s = [float(r["success_once"]) for r in rows]
    for i in range(len(s) - 2):
        if all(v >= 0.9 for v in s[i:i + 3]):
            return steps[i]
    last = (HALT - 1) // INTERVAL * INTERVAL
    if not steps or steps[-1] < last:
        raise SystemExit(f"{run}: final_ckpt.pt, no t90, and eval.csv ends at {steps[-1] if steps else 'nothing'}, "
                         f"not at the halt's last eval {last}")
    return HALT


def ranks_of(x):
    order = sorted(range(len(x)), key=lambda k: abs(x[k]))
    r, i = [0.0] * len(x), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and abs(x[order[j + 1]]) == abs(x[order[i]]):
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def signed_rank_p(d):
    """One-sided exact Wilcoxon signed-rank p for d > 0 (projection_test.py's): zeros dropped, average ranks, every
    sign flip."""
    x = [v for v in d if v != 0]
    if not x:
        return 1.0
    r = ranks_of(x)
    w = sum(rk for rk, v in zip(r, x) if v > 0)
    n = len(x)
    hits = sum(sum(r[k] for k in range(n) if m >> k & 1) >= w - 1e-9 for m in range(1 << n))
    return hits / (1 << n)


def w_null(d):
    """(observed 2*W+, Counter of 2*W+ over all sign flips) for one task's nonzero d."""
    x = [v for v in d if v != 0]
    r2 = [int(round(2 * rk)) for rk in ranks_of(x)]
    obs = sum(rk for rk, v in zip(r2, x) if v > 0)
    dist = Counter({0: 1})
    for rk in r2:
        nxt = Counter()
        for s, c in dist.items():
            nxt[s] += c
            nxt[s + rk] += c
        dist = nxt
    return obs, dist


def stratified_p(ds):
    """Sum over tasks of the per-task W+; exact one-sided p by convolving the per-task sign-flip distributions."""
    total, dist = 0, Counter({0: 1})
    for d in ds:
        obs, dd = w_null(d)
        total += obs
        nxt = Counter()
        for s1, c1 in dist.items():
            for s2, c2 in dd.items():
                nxt[s1 + s2] += c1 * c2
        dist = nxt
    n = sum(dist.values())
    return sum(c for s, c in dist.items() if s >= total) / n


def sign_test_p(k_pos, k):
    return sum(math.comb(k, j) for j in range(k_pos, k + 1)) / 2 ** k


def med_bounds(d):
    """Median of d with every open seed (None) at -inf and at +inf."""
    lo = [(-math.inf if v is None else v) for v in d]
    hi = [(math.inf if v is None else v) for v in d]
    return float(np.median(lo)), float(np.median(hi))


def fmt(v):
    return "  open" if v is None else ("  >8.0" if v >= HALT else f"{v / 1e6:6.2f}")


def task_table(runs, task, suite, arms):
    T = {}
    for a in arms:
        for s in SEEDS:
            r = runs / f"{task}_{a}_s{s}"
            T[a, s] = t90(r, want_args(task, a, s, suite), suite["env_kwargs"][task]) if r.exists() else None
    return T


def per_task(T, label, comps, out, key):
    """Rule 2 per comparator; returns {c: (d list, median or None, p or None, verdict, bounds, ratio)}."""
    res = {}
    print(f"\n{label}: t90 in M steps per seed (>8.0 = censored at the 8M halt; open = not finished)")
    for a in ARMS + [c for c in comps if c not in ARMS]:
        if (a, SEEDS[0]) not in T:
            continue
        vals = [T[a, s] for s in SEEDS]
        med = f"{np.median(vals) / 1e6:.2f}" if None not in vals else "open"
        print(f"  {a:9s} " + " ".join(fmt(v) for v in vals) + f"   median {med}")
    for c in comps:
        d = [None if T[c, s] is None or T["nsa64", s] is None else (T[c, s] - T["nsa64", s]) / INTERVAL
             for s in SEEDS]
        lo, hi = med_bounds(d)
        if None in d:
            res[c] = (d, None, None, "open", (lo, hi), None)
            print(f"  d({c} - nsa64): " + " ".join("open" if v is None else f"{v:g}" for v in d) + "  -> open")
            continue
        med, p_fast, p_slow = float(np.median(d)), signed_rank_p(d), signed_rank_p([-v for v in d])
        verdict = ("nsa64 faster" if med >= 1 and p_fast <= 0.05 else
                   f"{c} faster" if med <= -1 and p_slow <= 0.05 else "no difference shown")
        ratio = float(np.median([T[c, s] for s in SEEDS]) / np.median([T["nsa64", s] for s in SEEDS]))
        res[c] = (d, med, p_fast, verdict, (lo, hi), ratio)
        print(f"  d({c} - nsa64) in intervals: " + " ".join(f"{v:g}" for v in d) +
              f"   median {med:g}, signed-rank p {p_fast:.4f} (mirrored {p_slow:.4f}), median-t90 ratio "
              f"{ratio:.2f} -> {verdict}")
    out[key] = {c: {"d": r[0], "median": r[1], "p": r[2], "verdict": r[3]} for c, r in res.items()}
    return res


def rule3(res, comps, md5, label, deciding=True):
    """Rules 3a / 3b over the tasks in res; returns {c: (3a verdict, 3b verdict)}. deciding=False prints the numbers
    only (counted + near-transfer tasks: reported, never a verdict)."""
    K = len(res)
    verdicts = {}
    print(f"\nrule 3 over {label} (K = {K}):")
    if not deciding:
        for c in comps:
            full = [r[c] for r in res.values() if r[c][1] is not None]
            n_pos = sum(1 for r in full if r[1] >= 1)
            if len(full) == K and K:
                print(f"  vs {c:9s}: pooled signed-rank p {stratified_p([r[0] for r in full]):.2e}; tasks with median "
                      f"d >= 1: {n_pos} of {K}, sign-test p {sign_test_p(n_pos, K):.4f}")
            else:
                print(f"  vs {c:9s}: open")
        return verdicts
    for c in comps:
        full = [r[c] for r in res.values() if r[c][1] is not None]
        c_faster = [t for t, r in res.items() if r[c][3] == f"{c} faster"]
        if K <= 2:
            v3a = "not testable (K <= 2)"
        elif len(md5) > 1:
            v3a = "open (runs from different code versions)"
        elif c_faster:
            v3a = f"not shown ({c} faster on {', '.join(c_faster)})"
        elif len(full) < K:
            v3a = "open (unfinished runs)"
        else:
            sp = stratified_p([r[0] for r in full])
            v3a = f"FASTER ACROSS THE SUITE (pooled p {sp:.2e})" if sp <= 0.05 else f"not shown (pooled p {sp:.3f})"
        lo = sum(1 for r in res.values() if (r[c][1] if r[c][1] is not None else r[c][4][0]) >= 1)
        hi = sum(1 for r in res.values() if (r[c][1] if r[c][1] is not None else r[c][4][1]) >= 1)
        p_lo, p_hi = sign_test_p(lo, K) if K else 1.0, sign_test_p(hi, K) if K else 1.0
        if K <= 4:
            v3b = "untestable (K <= 4: even K/K gives p >= 1/16)"
        elif len(md5) > 1:
            v3b = "open (runs from different code versions)"
        elif p_lo <= 0.05:
            v3b = "GENERALISES"
        elif p_hi > 0.05:
            v3b = "not shown"
        else:
            v3b = "open"
        verdicts[c] = (v3a, v3b)
        print(f"  vs {c:9s}: 3a {v3a}")
        print(f"  {'':12s} 3b tasks with median d >= 1: {lo}" + (f"-{hi}" if hi != lo else "") + f" of {K}, "
              f"sign-test p {p_lo:.4f}" + (f"-{p_hi:.4f}" if hi != lo else "") + f" -> {v3b}")
        if len(full) == K and K:
            every = all(r[3] == "nsa64 faster" for r in full)
            gm = math.exp(np.mean([math.log(r[5]) for r in full]))
            print(f"  {'':12s} reported: nsa64 faster on every task (rule 2 on all {K}): {every}; geometric mean of "
                  f"median-t90 ratios {c} / nsa64 {gm:.2f}")
    return verdicts


def retention(ret_dir, logs_dir, runs, tasks):
    """Rules 5 and 6. Returns (exact verdict, {task: FT median})."""
    if not ret_dir:
        print("\nretention: no --ret given (rules 5b and 6 open)")
    fails, missing, ft = [], [], {}
    for task in tasks:
        ftv = []
        for s in SEEDS:
            for arm in ("nsa64", "monitor"):
                name = f"{task}_{arm}_s{s}"
                if not (runs / name / "final_ckpt.pt").exists():
                    missing.append(name)
                    continue
                old = expert = None
                f = Path(ret_dir) / f"{name}.json" if ret_dir else None
                if f is not None and f.exists():
                    rj = json.load(open(f))
                    expert = rj["old_expert"]
                    for ck, r in rj["runs"].items():
                        if ck.endswith(f"{name}/final_ckpt.pt"):
                            old = r["old_head"]
                if arm == "monitor":
                    if old is None:
                        missing.append(name + " (retention)")
                    else:
                        ftv.append(old)
                    continue
                gap = None
                lf = Path(logs_dir) / f"logs_{name}.txt" if logs_dir else None
                if lf is not None and lf.exists():
                    m = GAP_RE.findall(lf.read_text(errors="replace"))
                    gap = float(m[-1]) if m else None
                if gap is None or old is None:
                    missing.append(name + (" (gap)" if gap is None else "") + (" (retention)" if old is None else ""))
                    continue
                if gap > TOL_GAP or old < expert - TOL_EP - 1e-12:
                    fails.append(f"{name}: gap {gap:.1e}, old head {old:.3f} (expert {expert:.3f})")
        if ftv:
            ft[task] = float(np.median(ftv)) if len(ftv) == len(SEEDS) else None
    print("\nrule 5 (exact retention, every nsa64 run): " +
          ("FAIL: " + "; ".join(fails) if fails else "open (missing: " + ", ".join(missing[:8]) +
           (" ..." if len(missing) > 8 else "") + ")" if missing else "holds across the suite"))
    for task, m in ft.items():
        print(f"rule 6 {task}: full fine-tuning's Pick retention, median over seeds "
              + ("open" if m is None else f"{m:.3f} -> {'forgets' if m < 0.95 else 'does not forget (< 0.95 not met)'}"))
    return ("fail" if fails else "open" if missing else "holds"), ft


def chain_analysis(chain_dir, suite, single, md5):
    """Rule 7. chain_dir holds runs/, eval/, stopped/ and chain_tasks.txt (scripts/run_chain.sh); md5 is the
    single-step runs' code version, which every chain run must share."""
    order = suite["chain"]
    D = Path(chain_dir)
    if (D / "chain_tasks.txt").exists() and (D / "chain_tasks.txt").read_text().split() != order:
        raise SystemExit(f"{D} ran the chain {(D / 'chain_tasks.txt').read_text().split()}, not suite.json's {order}")
    pat = re.compile(r"^(chain64|chain64p|chain64npr|ftchain)-(\d+)-([a-z0-9]+)_s(\d+)$")
    print(f"\nchain {' -> '.join(['pick'] + order)}")
    C = {}
    for d in sorted((D / "runs").iterdir()) if (D / "runs").is_dir() else []:
        m = pat.match(d.name)
        if m:
            arm, k, task, s = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            if not 2 <= k <= len(order) or order[k - 1] != task:
                raise SystemExit(f"{d}: not step {k} of the registered chain {order}")
            if s in SEEDS:
                C[arm, k, s] = t90(d)
    for arm in CHAIN_ARMS:
        for s in SEEDS:
            C[arm, 1, s] = single.get((order[0], FIRST_ARM[arm], s))
    stopped = {f.stem: f.read_text().strip() for f in sorted((D / "stopped").glob("*.txt"))} \
        if (D / "stopped").is_dir() else {}
    for arm in CHAIN_ARMS:
        if not any((arm, 2, s) in C for s in SEEDS):
            continue
        print(f"  {arm}: t90 per step" + (" (7c: level k writes into the free width 256 - 8k)" if arm == "chain64p"
                                          else ""))
        for k, task in enumerate(order, 1):
            v = [C.get((arm, k, s)) for s in SEEDS]
            done = [x for x in v if x is not None]
            print(f"    {k} {task:8s}" + (f" free {256 - 8 * k:3d}" if arm == "chain64p" else "") + ": "
                  + " ".join(fmt(x) for x in v) + (f"   median {np.median(done) / 1e6:.2f}" if len(done) == len(v)
                                                    else "   open"))
        for s in SEEDS:
            if f"{arm}_s{s}" in stopped:
                print(f"    s{s} stopped: {stopped[f'{arm}_s{s}']}")
    print("  7b feature reuse (single-step nsa64 - chain64p, intervals; > 0 = the chain helps):")
    for k, task in enumerate(order[1:], 2):
        d = [None if C.get(("chain64p", k, s)) is None or single.get((task, "nsa64", s)) is None else
             (single[task, "nsa64", s] - C["chain64p", k, s]) / INTERVAL for s in SEEDS]
        if None in d:
            print(f"    {k} {task}: open")
        else:
            print(f"    {k} {task}: " + " ".join(f"{x:g}" for x in d) + f"   median {np.median(d):g}, "
                  f"signed-rank p {signed_rank_p(d):.4f}")
    evals = {}
    for f in sorted((D / "eval").glob("*.json")) if (D / "eval").is_dir() else []:
        m = re.match(r"^(chain64|chain64p|chain64npr|ftchain)_s(\d+)_L(\d+)(_own)?$", f.stem)
        if m and int(m.group(2)) in SEEDS:
            key = (m.group(1), int(m.group(2)), bool(m.group(4)))
            if key not in evals or int(m.group(3)) > evals[key][0]:
                evals[key] = (int(m.group(3)), f)
    bad, missing, learned, kept = [], [], 0, 0
    for arm in ("chain64", "chain64p"):
        for s in SEEDS:
            if (arm, s, False) not in evals:
                missing.append(f"{arm}_s{s}")
                continue
            L, f = evals[arm, s, False]
            for name, r in json.load(open(f))["skills"].items():
                if r["ref_success"] < LEARNED:
                    continue
                learned += 1
                if r["success"] < r["ref_success"] - TOL_EP - 1e-12 or r["max_gap"] > TOL_GAP:
                    bad.append(f"{arm} s{s} {name}: {r['success']:.3f} vs own {r['ref_success']:.3f}, "
                               f"gap {r['max_gap']:.1e}")
                else:
                    kept += 1
    cmd5 = {v for n, v in MD5.items() if pat.match(n)}
    if cmd5 - md5 or len(md5) > 1:
        print(f"  chain runs' code versions {sorted(cmd5)}, single-step runs' {sorted(md5)}: 7a open")
        missing.append("one code version")
    verdict = "fail" if bad else "open" if missing else "holds"
    print(f"  7a every learned skill through the last level (chain64, chain64p): {kept} of {learned} learned skills "
          "kept; " + ("FAIL: " + "; ".join(bad) if bad else f"open (no eval for {', '.join(missing)})" if missing
                      else "exact for every learned skill in every seed"))
    for arm, own in (("chain64npr", True), ("chain64npr", False), ("ftchain", False)):
        sk = {}
        for (a, s, o), (L, f) in evals.items():
            if a == arm and o == own:
                for name, r in json.load(open(f))["skills"].items():
                    sk.setdefault(name, []).append(r["success"] - r["ref_success"])
        if sk:
            print(f"  7d {arm} ({'own level' if own else 'last level'}): success minus its own network's, mean over "
                  "seeds: " + ", ".join(f"{n} {np.mean(v):+.3f}" for n, v in sk.items()))
    return verdict


def complete_seeds(runs, tasks, arms):
    """The largest m such that every run <task>_<arm>_s<seed <= m> is finished."""
    m = 0
    for s in range(1, 9):
        if all((runs / f"{t}_{a}_s{s}" / "final_ckpt.pt").exists() for t in tasks for a in arms):
            m = s
        else:
            break
    return m


def main():
    global SEEDS
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--ret", default="")
    p.add_argument("--logs", default="")
    p.add_argument("--dev", action="store_true", help="also report place and stack (never counted in rules 3-4)")
    p.add_argument("--fncl", action="store_true", help="also report the optional FN-PPO arm")
    p.add_argument("--chain", default="", help="scripts/run_chain.sh's CHAIN_DIR")
    p.add_argument("--cutoff", action="store_true", help="rule C: seeds 1..m, m the largest complete seed")
    p.add_argument("--out", default="")
    a = p.parse_args()
    runs, suite = Path(a.runs), json.load(open(a.suite))
    frozen = suite.get("protocol", {}).get("suite_test.py")
    if frozen and frozen != hashlib.md5(Path(__file__).read_bytes()).hexdigest():
        raise SystemExit("this analysis script is not the one frozen in suite.json (protocol.suite_test.py): no verdict")
    counted, near, dev = suite["counted"], suite["near"], suite["dev"]
    comps = COMPARATORS + (["fncl"] if a.fncl else [])
    if a.cutoff:
        m = complete_seeds(runs, counted + near + dev, ARMS)
        SEEDS = list(range(1, m + 1))
        print(f"rule C (cut-off): seeds 1..{m}" + ("   (m <= 4: rule 2 cannot reach p <= 0.05)" if m <= 4 else ""))
        if m == 0:
            raise SystemExit("no complete seed")
    print(f"suite (frozen {suite.get('written')}): counted {counted} (K = {len(counted)}), near {near}, dev {dev}")
    out, res_c, res_n, single = {}, {}, {}, {}
    T = task_table(runs, "place", suite, ["nsa64"])      # R0 first: nothing else is read if it fails
    if len(SEEDS) == 8 and all(T["nsa64", s] is not None for s in SEEDS):
        got = [T["nsa64", s] / INTERVAL for s in SEEDS]
        diff = max(abs(g - r) for g, r in zip(got, PLACE_PROTECT_NP))
        print(f"\nR0 regression, place nsa64 (intervals) {' '.join(f'{g:g}' for g in got)} vs runs/protect_np "
              f"{' '.join(map(str, PLACE_PROTECT_NP))}: " + ("identical" if diff == 0 else "within one interval"
              if diff <= 1 else "DIFFERENT"))
        if diff > 1:
            raise SystemExit("R0 failed: the suite's code changed PlaceCube's training; stop and find out why "
                             "(no other rule is read)")
    else:
        print("\nR0 regression: open (place_nsa64 seeds 1-8 not all finished)")
    for task in counted + near + (dev if a.dev else []):
        stratum = "counted" if task in counted else "near-transfer: reported" if task in near else \
            "development task: reported"
        T = task_table(runs, task, suite, ARMS + (["fncl"] if a.fncl else []))
        r = per_task(T, f"{task} ({stratum})", comps, out, task)
        (res_c if task in counted else res_n if task in near else {})[task] = r
        for arm in ARMS:
            for s in SEEDS:
                single[task, arm, s] = T[arm, s]
    if not a.dev:           # the chain's first steps and R0 read the dev tasks' runs all the same
        for task in dev:
            T = task_table(runs, task, suite, ARMS)
            for arm in ARMS:
                for s in SEEDS:
                    single[task, arm, s] = T[arm, s]
    md5 = {MD5[n] for n in MD5 if n.split("_")[0] in counted + near + dev}
    print(f"\ncode versions of the suite's finished runs: {sorted(md5)}")
    v_c = rule3(res_c, comps, md5, "the counted tasks (deciding)")
    if res_n:
        rule3({**res_c, **res_n}, comps, md5, "counted + near-transfer tasks (reported, no verdict)", deciding=False)
    name = {"nsa64npr": "the prefix sum (nsa64npr)", "monitor": "full fine-tuning (monitor)"}
    yes = {c: v_c.get(c, ("",))[0].startswith("FASTER") for c in name}
    opn = {c: v_c.get(c, ("",))[0].startswith("open") for c in name}
    gen = all(v_c.get(c, ("", ""))[1] == "GENERALISES" for c in name)
    if len(counted) <= 2:
        head = "not testable (K <= 2): per-task results only"
    elif all(yes.values()):
        head = "NSA64 IS FASTER ACROSS THE SUITE THAN BOTH THE PREFIX SUM AND FULL FINE-TUNING" + \
            (", AND THE SPEED-UP GENERALISES TO NEW TASKS" if gen else "")
    elif any(opn.values()):
        head = "open (" + "; ".join(f"{name[c]}: {'faster so far' if yes[c] else 'open' if opn[c] else 'not shown'}"
                                    for c in name) + ")"
    elif any(yes.values()):
        head = "nsa64 is faster across the suite than " + " and ".join(name[c] for c in name if yes[c]) + " only"
    else:
        head = "not shown"
    print(f"\nHEADLINE (rule 4): {head}")
    ex, ft = retention(a.ret, a.logs, runs, counted + near + dev)
    ch = chain_analysis(a.chain, suite, single, md5) if a.chain else "not run"
    print(f"\nparameters added per skill (rule 8): nsa64 {(42 + 256) * 64 + 64 + 64 * 256 + 256 + 256 * 8 + 8:,}, "
          f"head {256 * 8 + 8:,}, monitor 0 (Pick's actor: {42 * 256 + 256 + 2 * (256 * 256 + 256) + 256 * 8 + 8:,})")
    if a.out:
        json.dump({"seeds": SEEDS, "tasks": out, "rule3": v_c, "headline": head, "rule5": ex, "rule6": ft,
                   "rule7a": ch, "code_md5": sorted(md5)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
