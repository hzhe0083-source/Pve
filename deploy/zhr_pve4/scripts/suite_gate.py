"""The suite's task gates and the frozen task list (pre-registered 2026-11-01, before any suite run). CPU only.
Candidates are task keys of scripts/suite_tasks.sh (env id, env kwargs and --obs_adapter are read from there), in
three strata:
  counted  ycb pushng pullng liftpeg sphere   a new object (YCB shapes, a peg, a sphere) or a new contact mode (no
           grasp): scripts/suite_test.py rules 2-4; K = the number that pass both gates (at most 5)
  near     beside onpair    the 4 cm cube under a development task's success test (beside: PlaceCube's carry and
           release, the goal next to cube B; onpair: StackCube's test, cube C resting on A or B alone passes,
           stack_pyramid.py:140-180): rule 2 reported, never counted in rules 3-4
  dev      place stack      the development tasks: always in (R0, the chain's first steps), never counted
Gates, for every counted and near candidate, run in the order listed (the risky ones first):
  Z  zero-shot: the Pick expert through the task's map, scripts/nsa_eval.py --ckpt=$PICK
     --skills=<key>:<env>:$PICK:0 (256 envs, seed 0, deterministic, first episode per env), succeeds in <= 30% of
     episodes, so the task is a new skill and not one Pick already has.
  L  learnable within the budget: the pilot (seed 0, which no confirmatory analysis uses; nsa64 and monitor with the
     confirmatory flags, scripts/run_protect.sh with PROT_DIR=runs/suite_pilot) reaches t90 <= 4,000,000 in at least
     one of the two arms (half the 8M halt, so the slower arms still have room).
A candidate is in the suite iff it passes both; there are no fallbacks and no replacements. A candidate whose env
cannot be built on both hosts (e.g. no YCB assets, or the YCB rule keeps fewer than 6 models) goes in --unavailable
and counts as dropped, with the reason in suite.json. For the pilot only PASS / FAIL / PENDING is printed, never a t90
or which arm passed: until the confirmatory runs are analysed nobody looks at how the arms compare (the pilot run
directories are not to be opened either). The registered chain is CHAIN_ORDER kept to the dev tasks and the tasks in
the suite (dev and near first, the riskier counted tasks last, since a chain stops at a step that does not learn).
  before the first pilot:  python scripts/suite_gate.py --freeze=runs/suite/protocol.json
  after the gates:         python scripts/suite_gate.py --zeroshot=runs/suite_zs --pilot=runs/suite_pilot/runs \
                               [--unavailable=ycb:reason] [--write=runs/suite/suite.json]
--freeze records the md5 of this script, scripts/suite_test.py, scripts/suite_tasks.sh and the wrwl package (as
ppo.py's code_md5); --write refuses to freeze suite.json if any of them changed since, or if a candidate is undecided.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

INTERVAL, HALT, Z_MAX, L_MAX = 102_400, 8_000_000, 0.30, 4_000_000
COUNTED = ["ycb", "pushng", "pullng", "liftpeg", "sphere"]
NEAR = ["beside", "onpair"]
DEV = ["place", "stack"]
CHAIN_ORDER = ["place", "stack", "beside", "onpair", "sphere", "ycb", "liftpeg", "pushng", "pullng"]
HERE = Path(__file__).resolve().parent


def task_flags(key, sh=HERE / "suite_tasks.sh"):
    """(env id, env kwargs dict, obs_adapter or None) of `task_env key` in scripts/suite_tasks.sh."""
    out = subprocess.run(["bash", "-c", f'source "{sh}"; task_env {key}'], capture_output=True, text=True).stdout
    if out.strip() == "UNKNOWN" or "--env_id=" not in out:
        raise SystemExit(f"{key}: not a task of {sh}")
    env_id = re.search(r"--env_id=(\S+)", out).group(1)
    kw = re.search(r"--env_kwargs='([^']*)'", out)
    ad = re.search(r"--obs_adapter=(\w+)", out)
    return env_id, json.loads(kw.group(1)) if kw else {}, ad.group(1) if ad else None


def code_md5(root=HERE.parent):
    """ppo.py's code_md5: every module of the wrwl package (ppo.py:668-672)."""
    code = hashlib.md5()
    for src in sorted((root / "wrwl").rglob("*.py")):
        code.update(src.read_bytes())
    return code.hexdigest()


def protocol_md5():
    files = {"suite_gate.py": HERE / "suite_gate.py", "suite_test.py": HERE / "suite_test.py",
             "suite_tasks.sh": HERE / "suite_tasks.sh"}
    out = {k: hashlib.md5(f.read_bytes()).hexdigest() for k, f in files.items()}
    out["wrwl_code_md5"] = code_md5()
    return out


def t90(run, want):
    """(t90 or None, finished): None when finished without 3 consecutive evals >= 0.9 (censored) or not finished."""
    if not (run / "final_ckpt.pt").exists():
        return None, False
    j = json.load(open(run / "args.json"))
    bad = {k: (j.get(k), v) for k, v in want.items() if j.get(k) != v}
    if json.loads(j.get("env_kwargs", "{}")) != want_kwargs(run):
        bad["env_kwargs"] = (j.get("env_kwargs"), want_kwargs(run))
    if bad:
        raise SystemExit(f"{run}: args.json (found, expected) {bad}")
    rows = list(csv.DictReader(open(run / "eval.csv")))
    s = [float(r["success_once"]) for r in rows]
    for i in range(len(s) - 2):
        if all(v >= 0.9 for v in s[i:i + 3]):
            return int(float(rows[i]["global_step"])), True
    return None, True


def want_kwargs(run):
    return task_flags(run.name.split("_")[0])[1]


def gate_z(zs_dir, task):
    f = Path(zs_dir) / f"{task}.json"
    if not f.exists():
        return "PENDING", None
    s = json.load(open(f))["skills"][task]["success"]
    return ("PASS" if s <= Z_MAX else "FAIL"), s


def gate_l(pilot, task):
    env_id, _, adapter = task_flags(task)
    res = []
    for arm in ("nsa64", "monitor"):
        want = {"env_id": env_id, "obs_adapter": adapter, "seed": 0, "halt_at": HALT, "stop_success": 0.9,
                "stop_patience": 3, "num_envs": 1024, "num_steps": 50, "eval_freq": 2, "num_eval_envs": 256,
                "nullspace_adapter": 64 if arm == "nsa64" else 0,
                "protect_old": "monitor" if arm == "monitor" else "none"}
        res.append(t90(Path(pilot) / f"{task}_{arm}_s0", want))
    if any(t is not None and t <= L_MAX for t, _ in res):
        return "PASS"
    return "FAIL" if all(fin for _, fin in res) else "PENDING"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", default="", help="write the protocol files' md5 here (before the first pilot)")
    p.add_argument("--zeroshot", default="", help="nsa_eval.py outputs, <task>.json")
    p.add_argument("--pilot", default="", help="the pilot queue's runs/ directory")
    p.add_argument("--unavailable", default="", help="task:reason,... whose env cannot be built (dropped)")
    p.add_argument("--write", default="", help="freeze suite.json here (next to the --freeze file)")
    p.add_argument("--protocol", default="", help="the --freeze file (default: protocol.json next to --write)")
    a = p.parse_args()
    if a.freeze:
        if Path(a.freeze).exists():
            raise SystemExit(f"{a.freeze} exists: the protocol is frozen once")
        Path(a.freeze).parent.mkdir(parents=True, exist_ok=True)
        out = {**protocol_md5(), "counted": COUNTED, "near": NEAR, "dev": DEV, "chain_order": CHAIN_ORDER,
               "frozen": time.strftime("%F %T")}
        json.dump(out, open(a.freeze, "w"), indent=1)
        print(f"froze {a.freeze}: {out}")
        return
    if not (a.zeroshot and a.pilot):
        raise SystemExit("--zeroshot and --pilot, or --freeze")
    gone = dict(x.split(":", 1) for x in a.unavailable.split(",") if x)
    status, zs = {}, {}
    for task in COUNTED + NEAR:
        if task in gone:
            print(f"  {task:8s} unavailable ({gone[task]}): dropped")
            status[task] = "DROPPED"
            continue
        z, zval = gate_z(a.zeroshot, task)
        zs[task] = zval
        if z == "PENDING":
            print(f"  {task:8s} Z PENDING")
            status[task] = "PENDING"
            continue
        if z == "FAIL":
            print(f"  {task:8s} Z FAIL (Pick's own head succeeds in {zval:.3f} > {Z_MAX}): dropped")
            status[task] = "DROPPED"
            continue
        ell = gate_l(a.pilot, task)
        print(f"  {task:8s} Z PASS ({zval:.3f})   L {ell}")
        status[task] = {"PASS": "SELECTED", "FAIL": "DROPPED", "PENDING": "PENDING"}[ell]
    counted = [t for t in COUNTED if status[t] == "SELECTED"]
    near = [t for t in NEAR if status[t] == "SELECTED"]
    decided = all(v != "PENDING" for v in status.values())
    print(f"\ncounted so far: {counted} (K = {len(counted)}); near-transfer: {near}"
          + ("" if decided else "   (not final: a candidate is pending)"))
    if decided and len(counted) < 3:
        print("  K < 3: rule 3 cannot be tested; per-task results only (suite_test.py)")
    elif decided and len(counted) < 5:
        print("  K < 5: rule 3a (pooled) can decide; rule 3b (sign test, 'generalises to new tasks') is untestable")
    if a.write:
        if not decided:
            raise SystemExit("not written: the suite is not decided yet")
        if Path(a.write).exists():
            raise SystemExit(f"{a.write} exists: the suite is frozen once")
        pf = Path(a.protocol) if a.protocol else Path(a.write).parent / "protocol.json"
        if not pf.exists():
            raise SystemExit(f"{pf} missing: --freeze was not run before the pilots")
        frozen, now = json.load(open(pf)), protocol_md5()
        changed = {k: (frozen.get(k), v) for k, v in now.items() if frozen.get(k) != v}
        if changed:
            raise SystemExit(f"not written: changed since --freeze {changed}")
        md5 = set()
        for t in counted + near:
            for arm in ("nsa64", "monitor"):
                f = Path(a.pilot) / f"{t}_{arm}_s0" / "args.json"
                if f.exists():
                    md5.add(json.load(open(f)).get("code_md5"))
        if md5 and md5 != {now["wrwl_code_md5"]}:
            raise SystemExit(f"not written: the pilots ran code {sorted(map(str, md5))}, this checkout is "
                             f"{now['wrwl_code_md5']}")
        tasks = counted + near + DEV
        flags = {t: task_flags(t) for t in tasks}
        out = {"counted": counted, "near": near, "dev": DEV,
               "env": {t: flags[t][0] for t in tasks}, "env_kwargs": {t: flags[t][1] for t in tasks},
               "obs_adapter": {t: flags[t][2] for t in tasks},
               "chain": [t for t in CHAIN_ORDER if t in tasks],
               "zeroshot": {t: zs.get(t) for t in COUNTED + NEAR}, "status": status, "unavailable": gone,
               "pilot_code_md5": sorted(map(str, md5)), "protocol": frozen, "written": time.strftime("%F %T")}
        json.dump(out, open(a.write, "w"), indent=1)
        print(f"wrote {a.write}: counted {counted}, near {near}, chain {out['chain']}")


if __name__ == "__main__":
    main()
