"""Synthetic-run tests for scripts/suite_test.py and scripts/suite_gate.py (CPU, numpy and scipy; no simulator, no
GPU). Each scenario writes run directories as scripts/run_protect.sh and scripts/run_chain.sh leave them (args.json
with the queue's flags for the task as scripts/suite_tasks.sh gives them, eval.csv every 102,400 steps, final_ckpt.pt,
the chain's eval/ and stopped/), retention JSONs as retention_eval.py writes them and ppo.py's end-of-run gap line,
then checks the verdict lines. The exact tests are checked against scipy.
  python scripts/test_suite_test.py
"""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import suite_gate as SG  # noqa: E402
import suite_test as ST  # noqa: E402

INTERVAL, HALT = 102_400, 8_000_000
FAILS = []
FLAGS = {}


def flags(task):
    if task not in FLAGS:
        FLAGS[task] = SG.task_flags(task)
    return FLAGS[task]


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def suite_json(counted, near, dev=("place", "stack")):
    tasks = list(counted) + list(near) + list(dev)
    return {"counted": list(counted), "near": list(near), "dev": list(dev),
            "env": {t: flags(t)[0] for t in tasks}, "env_kwargs": {t: flags(t)[1] for t in tasks},
            "obs_adapter": {t: flags(t)[2] for t in tasks},
            "chain": [t for t in SG.CHAIN_ORDER if t in tasks], "written": "test"}


def eval_rows(t90_steps):
    if t90_steps is None or t90_steps >= HALT:          # censored: evals to the halt's last one, never 3 >= 0.9
        return [(k * INTERVAL, 0.5) for k in range((HALT - 1) // INTERVAL + 1)]
    k0 = t90_steps // INTERVAL
    return [(k * INTERVAL, 0.4) for k in range(k0)] + [((k0 + i) * INTERVAL, 0.95) for i in range(3)]


def write_eval(d, t90_steps):
    with open(d / "eval.csv", "w") as f:
        f.write("iteration,global_step,success_once,success_at_end,return,prior_coef,wall_s\n")
        for i, (st, s) in enumerate(eval_rows(t90_steps)):
            f.write(f"{2 * i + 1},{st},{s},0,0,0,0\n")


def make_run(root, suite, task, arm, seed, t90_steps, md5="abc", finished=True, gap=1.2e-6, old=0.992, ret=None,
             args_override=None):
    name = f"{task}_{arm}_s{seed}"
    d = root / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    args = ST.want_args(task, arm, seed, suite)
    args.update({"exp_name": name, "code_md5": md5, "env_kwargs": json.dumps(suite["env_kwargs"][task]),
                 **(args_override or {})})
    json.dump(args, open(d / "args.json", "w"))
    write_eval(d, t90_steps)
    if finished:
        (d / "final_ckpt.pt").write_bytes(b"x")
    if arm in ("nsa64", "nsa64npr"):
        (root / f"logs_{name}.txt").write_text(f"...\nnullspace adapter: max |old head on h~ - old head on h| = "
                                               f"{gap:.2e}\ndone {name}\n")
    if ret is not None:
        ret.mkdir(exist_ok=True)
        json.dump({"old_expert": 0.992, "runs": {f"/x/runs/{name}/final_ckpt.pt": {"old_head": old}}},
                  open(ret / f"{name}.json", "w"))


def build(root, suite, t90s, over=None, seeds=range(1, 9)):
    """t90s[task][arm] = list of 8 t90s in intervals (None = censored)."""
    ret = root / "ret"
    for t, arms in t90s.items():
        for arm, vals in arms.items():
            for s in seeds:
                v = vals[s - 1]
                make_run(root, suite, t, arm, s, HALT if v is None else v * INTERVAL, ret=ret,
                         old=(0.80 if arm == "monitor" else 0.992), **(over or {}).get((t, arm, s), {}))
    json.dump(suite, open(root / "suite.json", "w"))


def run(root, *extra):
    cmd = [sys.executable, str(HERE / "suite_test.py"), f"--runs={root}/runs", f"--suite={root}/suite.json",
           f"--ret={root}/ret", f"--logs={root}", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


def faster(base, delta, rng):
    return [b + delta + rng.choice([0, 0, 1]) for b in base]


def t90s_for(tasks, rng, mon_delta=2, npr_delta=3):
    out = {}
    for t in tasks:
        base = [rng.randint(5, 20) for _ in range(8)]
        out[t] = {"nsa64": base, "nsa64npr": faster(base, npr_delta, rng), "monitor": faster(base, mon_delta, rng),
                  "head": faster(base, 10, rng)}
    return out


def scenario(tmp, name, counted, near=(), dev_too=True, seed=1, mutate=None, over=None, seeds=range(1, 9)):
    rng = random.Random(seed)
    suite = suite_json(counted, near)
    t = t90s_for(list(counted) + list(near) + (["place", "stack"] if dev_too else []), rng)
    if dev_too:
        t["place"]["nsa64"] = [6, 7, 6, 6, 6, 6, 7, 6]
    if mutate:
        mutate(t)
    root = Path(tmp) / name
    build(root, suite, t, over, seeds)
    return root, suite, t


def make_chain(root, suite, arms=("chain64", "chain64p"), seeds=range(1, 9), md5="abc", tweak=None, stop=None):
    """A finished chain directory as scripts/run_chain.sh leaves it; tweak(arm, s, skills) edits an eval."""
    D = root / "chain"
    order = suite["chain"]
    (D / "eval").mkdir(parents=True, exist_ok=True)
    (D / "stopped").mkdir(exist_ok=True)
    (D / "chain_tasks.txt").write_text(" ".join(order) + "\n")
    for arm in arms:
        for s in seeds:
            for k, t in enumerate(order[1:], 2):
                d = D / "runs" / f"{arm}-{k}-{t}_s{s}"
                d.mkdir(parents=True)
                json.dump({"code_md5": md5}, open(d / "args.json", "w"))
                write_eval(d, 6 * INTERVAL)
                (d / "final_ckpt.pt").write_bytes(b"x")
            sk = {n: {"success": 0.99, "ref_success": 0.99, "max_gap": 1e-6} for n in ["pick"] + order}
            if tweak:
                tweak(arm, s, sk)
            json.dump({"skills": sk}, open(D / "eval" / f"{arm}_s{s}_L{len(order)}.json", "w"))
    if stop:
        (D / "stopped" / f"{stop[0]}.txt").write_text(stop[1] + "\n")
    return D


def gate_run(gate, zs, pil, *extra):
    g = subprocess.run([sys.executable, str(HERE / "suite_gate.py"), f"--zeroshot={zs}", f"--pilot={pil}", *extra],
                       capture_output=True, text=True)
    return g.stdout + g.stderr, g.returncode


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1 exact tests against scipy (no ties, no zeros)
        rng = random.Random(0)
        worst = 0.0
        for _ in range(200):
            d = [v * rng.choice([-1, 1]) for v in rng.sample(range(1, 31), 8)]
            worst = max(worst, abs(ST.signed_rank_p(d) - wilcoxon(d, alternative="greater", method="exact").pvalue))
        check(worst < 1e-12, f"signed-rank p equals scipy's exact Wilcoxon (largest difference {worst:.1e})")
        check(all(abs(ST.sign_test_p(k, K) - binomtest(k, K, 0.5, alternative="greater").pvalue) < 1e-12
                  for K in range(1, 11) for k in range(K + 1)), "sign-test p equals scipy's binomtest")
        d1 = [3, 2, 3, 3, -1, 3, 5, 2]
        check(abs(ST.stratified_p([d1]) - ST.signed_rank_p(d1)) < 1e-12, "stratified p with one task = signed-rank p")
        d2 = [1, -2, 4, 0, 2, 2, -1, 3]
        x1, x2 = [v for v in d1 if v], [v for v in d2 if v]
        r1, r2 = ST.ranks_of(x1), ST.ranks_of(x2)
        w = sum(r for r, v in zip(r1, x1) if v > 0) + sum(r for r, v in zip(r2, x2) if v > 0)
        n1, n2 = len(x1), len(x2)
        hits = sum(1 for m1 in range(1 << n1) for m2 in range(1 << n2)
                   if sum(r1[k] for k in range(n1) if m1 >> k & 1) + sum(r2[k] for k in range(n2) if m2 >> k & 1)
                   >= w - 1e-9)
        check(abs(ST.stratified_p([d1, d2]) - hits / 2 ** (n1 + n2)) < 1e-12, "stratified p equals brute force")
        check(SG.task_flags("sphere") == ("PlaceSphere-v1", {"robot_uids": "panda_wristcam", "reward_mode": "sparse"},
                                          "sphere_to_pick") and SG.task_flags("place")[2] is None
              and SG.task_flags("stack") == ("StackCubeNarrow-v1", {"reward_mode": "sparse"}, "stack_to_pick"),
              "suite_gate reads env id, kwargs and adapter from suite_tasks.sh")
        check(set(SG.COUNTED + SG.NEAR + SG.DEV) == set(SG.CHAIN_ORDER) and not set(SG.COUNTED) & set(SG.NEAR),
              "strata partition the registered chain order")

        # 2 K = 5, nsa64 faster everywhere: 3a and 3b hold for both comparators, R0, rules 5 and 6
        C5 = ["ycb", "pushng", "pullng", "liftpeg", "sphere"]
        root, suite, _ = scenario(tmp, "all5", C5, ["beside", "onpair"])
        out, rc = run(root, "--dev")
        check(rc == 0, "K=5 all faster: runs")
        check("HEADLINE (rule 4): NSA64 IS FASTER ACROSS THE SUITE THAN BOTH THE PREFIX SUM AND FULL FINE-TUNING, AND "
              "THE SPEED-UP GENERALISES TO NEW TASKS" in out, "K=5 all faster: pooled and sign test, both comparators")
        check("sign-test p 0.0312" in out, "K=5: sign-test p = 1/32")
        check("R0 regression, place nsa64 (intervals) 6 7 6 6 6 6 7 6" in out and "identical" in out, "R0 identical")
        check("holds across the suite" in out, "K=5: rule 5 holds")
        check("forgets" in out and "0.800" in out, "rule 6 reports full fine-tuning forgetting (0.80 < 0.95)")
        check("onpair (near-transfer: reported)" in out and "counted + near-transfer tasks (reported, no verdict)" in out,
              "near-transfer tasks reported in their own stratum")
        # R0 that does not reproduce protect_np stops everything else
        def r0_off(t):
            t["place"]["nsa64"] = [6, 7, 6, 6, 9, 6, 7, 6]
        root, *_ = scenario(tmp, "r0bad", C5[:3], mutate=r0_off)
        out, rc = run(root)
        check(rc != 0 and "R0 failed" in out and "rule 3" not in out, "R0 off by 3 intervals: stop before any rule")
        # 3 K = 4: 3a decides, 3b untestable
        root, *_ = scenario(tmp, "all4", C5[:4])
        out, _ = run(root)
        check("HEADLINE (rule 4): NSA64 IS FASTER ACROSS THE SUITE THAN BOTH THE PREFIX SUM AND FULL FINE-TUNING\n"
              in out and "untestable (K <= 4" in out, "K=4: pooled test decides, sign test untestable")

        # 4 K = 3, one task without a speed-up against monitor (d all 0): still pooled-significant, no task where
        #   monitor is faster -> 3a holds against monitor; 3b untestable
        def flat_monitor(t):
            t["pushng"]["monitor"] = list(t["pushng"]["nsa64"])
        root, *_ = scenario(tmp, "k3flat", C5[:3], mutate=flat_monitor)
        out, _ = run(root)
        check("vs monitor  : 3a FASTER ACROSS THE SUITE" in out, "K=3, one task d = 0 vs monitor: 3a still holds")

        # 5 K = 3, monitor significantly faster on one task -> 3a not shown against monitor
        def monitor_wins(t):
            t["pushng"]["monitor"] = [v - 3 for v in t["pushng"]["nsa64"]]
        root, *_ = scenario(tmp, "k3lose", C5[:3], mutate=monitor_wins)
        out, _ = run(root)
        check("vs monitor  : 3a not shown (monitor faster on pushng)" in out
              and "nsa64 is faster across the suite than the prefix sum (nsa64npr) only" in out,
              "a task where monitor is faster: 3a fails against monitor only")

        # 6 K = 2 -> not testable
        root, *_ = scenario(tmp, "k2", C5[:2])
        out, _ = run(root)
        check("not testable (K <= 2)" in out and "HEADLINE (rule 4): not testable" in out, "K=2: not testable")

        # 7 open runs -> open; near-transfer losses do not touch the counted verdict
        def near_loses(t):
            t["beside"]["monitor"] = [v - 3 for v in t["beside"]["nsa64"]]
        root, *_ = scenario(tmp, "open", C5[:3], ["beside"], mutate=near_loses,
                            over={("ycb", "monitor", s): {"finished": False} for s in (1, 2)})
        out, _ = run(root)
        check("vs monitor  : 3a open (unfinished runs)" in out and "HEADLINE (rule 4): open (the prefix sum (nsa64npr): "
              "faster so far; full fine-tuning (monitor): open)" in out, "open counted runs: 3a and the headline open")
        check("beside (near-transfer: reported)" in out and "monitor faster" in out,
              "near-transfer task where monitor wins is reported")

        # 8 retention leak, bad args, code versions
        root, *_ = scenario(tmp, "leak", C5[:3], over={("ycb", "nsa64", 3): {"gap": 1e-3}})
        out, _ = run(root)
        check("rule 5 (exact retention, every nsa64 run): FAIL: ycb_nsa64_s3" in out, "a leaking run fails rule 5")
        root, *_ = scenario(tmp, "badargs", C5[:3], over={("pullng", "head", 4): {"args_override": {"halt_at": 3_000_000}}})
        out, rc = run(root)
        check(rc != 0 and "pullng_head_s4: args.json" in out, "a run with other flags stops the analysis")
        root, *_ = scenario(tmp, "badadapter", C5[:3],
                            over={("pushng", "nsa64", 2): {"args_override": {"obs_adapter": None}}})
        out, rc = run(root)
        check(rc != 0 and "pushng_nsa64_s2: args.json" in out, "a run without the task's adapter stops the analysis")
        root, *_ = scenario(tmp, "badkw", C5[:3],
                            over={("ycb", "head", 1): {"args_override": {"env_kwargs": '{"reward_mode": "sparse"}'}}})
        out, rc = run(root)
        check(rc != 0 and "ycb_head_s1: args.json" in out and "env_kwargs" in out, "other env kwargs stop it")
        root, *_ = scenario(tmp, "md5", C5[:3], over={("pullng", "head", 4): {"md5": "other"}})
        out, _ = run(root)
        check("3a open (runs from different code versions)" in out, "two code versions: rule 3 open")

        # 9 censored comparators count at the halt
        def censor_head(t):
            t["ycb"]["head"] = [None] * 8
        root, *_ = scenario(tmp, "cens", C5[:3], mutate=censor_head)
        out, _ = run(root)
        check(">8.0" in out, "censored runs shown as >8.0 and counted at the halt")

        # 10 the chain: 7a holds; unlearned skills are not counted; a learned skill's drop fails it
        root, suite, _ = scenario(tmp, "chain", C5[:3], ["beside", "onpair"])
        order = suite["chain"]
        check(order == ["place", "stack", "beside", "onpair", "ycb", "pushng", "pullng"], f"registered chain {order}")

        def unlearned(arm, s, sk):
            sk["pushng"] = {"success": 0.0, "ref_success": 0.2, "max_gap": 1e-6}
        make_chain(root, suite, tweak=unlearned, stop=("chain64p_s4", "step 7 pushng (chain64p-7-pushng_s4) has no 3 "
                                                                        "evals >= 0.9"))
        out, _ = run(root, f"--chain={root}/chain")
        n_learned = 2 * 8 * len(order)        # pick + the chain's skills, less the unlearned pushng, per eval
        check(f"7a every learned skill through the last level (chain64, chain64p): {n_learned} of {n_learned} learned "
              "skills kept; exact for every learned skill in every seed" in out,
              "chain: unlearned skills are not counted, 7a holds")
        check("s4 stopped: step 7 pushng" in out, "chain: a stopped chain is reported")
        check("free 240" in out and out.count("free 2") == len(order), "chain: the free width is shown for chain64p only")
        shutil.rmtree(root / "chain")

        def drop(arm, s, sk):
            if arm == "chain64p" and s == 5:
                sk["stack"]["success"] = 0.99 - 3 / 256
        make_chain(root, suite, tweak=drop)
        out, _ = run(root, f"--chain={root}/chain")
        check("FAIL: chain64p s5 stack" in out, "chain: a 3-episode drop of a learned skill fails 7a")
        shutil.rmtree(root / "chain")
        make_chain(root, suite, md5="other")
        out, _ = run(root, f"--chain={root}/chain")
        check("7a open" in out or "chain runs' code versions" in out, "chain runs from another code version: 7a open")
        shutil.rmtree(root / "chain")
        D = make_chain(root, suite)
        (D / "chain_tasks.txt").write_text("place stack sphere\n")
        out, rc = run(root, f"--chain={root}/chain")
        check(rc != 0 and "not suite.json's" in out, "chain: another chain order stops the analysis")

        # 11 cut-off: seeds 7 and 8 unfinished for one arm -> m = 6
        root, *_ = scenario(tmp, "cut", C5[:4], over={("liftpeg", "head", s): {"finished": False} for s in (7, 8)})
        out, _ = run(root, "--cutoff")
        check("rule C (cut-off): seeds 1..6" in out and "3a FASTER ACROSS THE SUITE" in out and "R0 regression: open" in out,
              "cut-off keeps seeds 1-6 for every task and arm")

        # 12 the gate: freeze, strata, Z / L, pending, unavailable, blindness, refusal after a change
        gate = Path(tmp) / "gate"
        zs, pil = gate / "zs", gate / "pilot" / "runs"
        zs.mkdir(parents=True)
        frozen = gate / "suite" / "protocol.json"
        g = subprocess.run([sys.executable, str(HERE / "suite_gate.py"), f"--freeze={frozen}"], capture_output=True,
                           text=True)
        check(g.returncode == 0 and frozen.exists(), "gate: --freeze writes protocol.json")
        md5 = json.load(open(frozen))["wrwl_code_md5"]
        z = {"ycb": 0.45, "pushng": 0.0, "pullng": 0.0, "liftpeg": 0.02, "sphere": 0.05, "beside": 0.0, "onpair": 0.0}
        for t, v in z.items():
            json.dump({"skills": {t: {"success": v}}}, open(zs / f"{t}.json", "w"))
        pt90 = {"pushng": (None, None), "pullng": (30, None), "liftpeg": (25, 39), "sphere": (8, 9), "beside": (6, 7),
                "onpair": (12, 13)}
        for t, (a1, a2) in pt90.items():
            env_id, kw, ad = SG.task_flags(t)
            for arm, v in (("nsa64", a1), ("monitor", a2)):
                d = pil / f"{t}_{arm}_s0"
                d.mkdir(parents=True)
                json.dump({"env_id": env_id, "env_kwargs": json.dumps(kw), "obs_adapter": ad, "seed": 0,
                           "halt_at": HALT, "stop_success": 0.9, "stop_patience": 3, "num_envs": 1024, "num_steps": 50,
                           "eval_freq": 2, "num_eval_envs": 256, "nullspace_adapter": 64 if arm == "nsa64" else 0,
                           "protect_old": "monitor" if arm == "monitor" else "none", "code_md5": md5},
                          open(d / "args.json", "w"))
                write_eval(d, None if v is None else v * INTERVAL)
                (d / "final_ckpt.pt").write_bytes(b"x")
        o, rc = gate_run(gate, zs, pil, f"--write={gate}/suite/suite.json")
        sj = json.load(open(gate / "suite" / "suite.json")) if (gate / "suite" / "suite.json").exists() else {}
        check(sj.get("counted") == ["pullng", "liftpeg", "sphere"] and sj.get("near") == ["beside", "onpair"],
              f"gate: counted {sj.get('counted')}, near {sj.get('near')} (ycb fails Z, pushng fails L)")
        check(sj.get("chain") == ["place", "stack", "beside", "onpair", "sphere", "liftpeg", "pullng"]
              and sj.get("obs_adapter", {}).get("liftpeg") == "peg_to_pick", f"gate: chain {sj.get('chain')}")
        lines = [ln for ln in o.splitlines() if " L " in ln]
        import re
        check(len(lines) == 6 and all(re.search(r" L (PASS|FAIL|PENDING)$", ln) for ln in lines),
              "gate prints only PASS / FAIL / PENDING for the pilot (no t90, no arm)")
        print("      " + "\n      ".join(o.strip().splitlines()))
        o, rc = gate_run(gate, zs, pil, f"--write={gate}/suite/suite.json")
        check(rc != 0 and "frozen once" in o, "gate: suite.json is written once")
        sj["protocol"]["suite_test.py"] = "0" * 32       # the analysis refuses a script other than the frozen one
        (gate / "a").mkdir()
        json.dump(sj, open(gate / "a" / "suite.json", "w"))
        p = subprocess.run([sys.executable, str(HERE / "suite_test.py"), f"--runs={gate}/a/runs", f"--suite={gate}/a/suite.json"],
                           capture_output=True, text=True)
        check(p.returncode != 0 and "not the one frozen" in p.stdout + p.stderr, "analysis: refuses an unfrozen script")
        (zs / "sphere.json").unlink()
        o, rc = gate_run(gate, zs, pil, f"--write={gate}/suite2/suite.json", f"--protocol={frozen}")
        check(rc != 0 and "not final" in o and "not decided" in o, "gate: a pending candidate blocks the write")
        o, rc = gate_run(gate, zs, pil, "--unavailable=sphere:no assets on haha-d")
        check("unavailable (no assets on haha-d): dropped" in o and "not final" not in o,
              "gate: an unavailable candidate is dropped and does not block")
        json.dump({"skills": {"sphere": {"success": 0.05}}}, open(zs / "sphere.json", "w"))
        bad = json.load(open(frozen))
        bad["suite_test.py"] = "0" * 32
        (gate / "suite3").mkdir()
        json.dump(bad, open(gate / "suite3" / "protocol.json", "w"))
        o, rc = gate_run(gate, zs, pil, f"--write={gate}/suite3/suite.json")
        check(rc != 0 and "changed since --freeze" in o, "gate: refuses to write after a protocol file changed")
        for f in pil.glob("*/args.json"):
            j = json.load(open(f))
            j["code_md5"] = "other"
            json.dump(j, open(f, "w"))
        (gate / "suite4").mkdir()
        shutil.copy(frozen, gate / "suite4" / "protocol.json")
        o, rc = gate_run(gate, zs, pil, f"--write={gate}/suite4/suite.json")
        check(rc != 0 and "the pilots ran code" in o, "gate: refuses pilots from another code version")
    print("\nFAILED: " + "; ".join(FAILS) if FAILS else "\nall tests passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
