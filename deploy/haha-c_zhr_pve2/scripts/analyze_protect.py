"""Summarise the --protect_old arms (scripts/run_protect.sh): per arm and seed, Stack steps to 90% (start of the first
run of 3 consecutive evals >= 0.9, as in the doc), the old skill's drift (mean ||(mu_old_head(new trunk) - mu_A)/sigma||^2
on 2000 of A's states, logged in protect.csv) at the end and at fixed steps, the mean share of each Adam step removed by
the projection (for monitor: the share a block projection WOULD remove, i.e. rho_B per step), and, if --ret is given, the
PickCube retention from scripts/retention_eval.py (old head on the new trunk): on final_ckpt.pt, and on ckpt_<t90>.pt
when a run saved a checkpoint at every eval (run_protect.sh's PROT_STOP for the 3M runs) and retention_eval.py evaluated
it (an extra column). fn runs whose Gram could not be factored in some iterations are listed below the table.
  python scripts/analyze_protect.py --runs=DIR/runs [--ret=ret.json] [--out=x.json]
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True)
p.add_argument("--ret", default="", help="comma-separated retention_eval.py outputs")
p.add_argument("--at", default="0.5e6,1e6,1.5e6")
p.add_argument("--out", default="")
a = p.parse_args()
AT = [float(x) for x in a.at.split(",")]


def rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def steps_to_90(ev):
    s = [float(r["success_once"]) for r in ev]
    for i in range(len(s) - 2):
        if min(s[i:i + 3]) >= 0.9:
            return float(ev[i]["global_step"])
    return np.inf


def at_step(pr, key, step):
    """Value of pr[key] at the last logged iteration with global_step <= step (nan if the run ended before)."""
    v = [float(r[key]) for r in pr if float(r["global_step"]) <= step]
    return v[-1] if v and float(pr[-1]["global_step"]) >= step else np.nan


ret = {}
for f in filter(None, a.ret.split(",")):
    for c, r in json.load(open(f))["runs"].items():
        ret[Path(c).parent.name if c.endswith("final_ckpt.pt") else Path(c).parent.name + "/" + Path(c).stem] = r

res = defaultdict(dict)
for d in sorted(Path(a.runs).iterdir()):
    m = re.match(r"(stack|place)_(\w+?)_s(\d+)$", d.name)
    if not m:
        continue
    arm, seed = (m.group(2) if m.group(1) == "stack" else f"{m.group(1)}:{m.group(2)}"), int(m.group(3))
    ev, pr, oe = rows(d / "eval.csv"), rows(d / "protect.csv"), rows(d / "old_eval.csv")
    if not ev:      # chain runs (scripts/run_chain.sh) write neither protect.csv nor old_eval.csv
        continue
    done = (d / "final_ckpt.pt").exists()
    lay = [k for k in pr[0] if k.startswith("removed")] if pr else []
    t90 = steps_to_90(ev)
    res[arm][seed] = {
        "t90": t90, "last_step": float(ev[-1]["global_step"]), "last_eval": float(ev[-1]["success_once"]),
        "drift_end": float(pr[-1]["old_drift"]) if pr else np.nan,
        "drift_at": [at_step(pr, "old_drift", s) if pr else np.nan for s in AT],
        # nanmean: fn logs nan while nothing is projected (--fn_warmup); the same as mean for the other arms
        "removed": [float(np.nanmean([float(r[k]) for r in pr if r["iteration"] != "0"])) for k in lay] if lay else [np.nan] * 3,
        # retention_eval.py (seed-0 starts) if it could load the run, else old_eval.csv's last row (the same starts)
        "pick_old_head": ret.get(d.name, {}).get("old_head", float(oe[-1]["old_success"]) if oe and done else np.nan),
        # with a checkpoint at every eval, ckpt_<t90>.pt holds the weights eval.csv scored at t90
        "pick_t90": ret.get(f"{d.name}/ckpt_{t90:.0f}", {}).get("old_head", np.nan) if np.isfinite(t90) else np.nan,
        # fn: iterations whose Gram could not be factored (fn.csv skipped; the nanmean above hides them)
        "fn_skipped": sum(int(r["skipped"]) for r in rows(d / "fn.csv")),
        "done": done,
    }


def key(arm):
    """Stack arms first, then other tasks' (task:arm), each in ORDER."""
    a = arm.split(":")[-1]
    return ":" in arm, ORDER.index(a) if a in ORDER else 99


ORDER = ["head", "monitor", "block09", "block095", "block099", "block095r", "gpm095", "gpm099",
         "block09g", "block095g", "block099g", "gpm095g",
         "distill001", "distill01", "distill1", "ewc10", "ewc100", "ewc1000", "l2_1", "l2_10", "l2_100",
         "distill1d", "klcl1", "klcl3", "fnbank", "fncl", "fncl_w5", "nsa16", "nsa64", "nsa256",
         "nsa16np", "nsa64np", "nsa256np", "chain64", "chain64p", "chain64np", "ftchain"]
T90 = any(np.isfinite(S[s]["pick_t90"]) for S in res.values() for s in S)     # some run was evaluated at t90
hdr = (f"{'arm':10s} | seeds | new skill t90 per seed | mean  | median | drift end | drift at " + "/".join(f"{s / 1e6:g}M" for s in AT)
       + " | removed per step L1/L2/L3 | " + ("Pick (old head) at the end | at t90" if T90 else "Pick (old head)"))
print(hdr)
print("-" * len(hdr))
for arm in sorted(res, key=key):
    S = res[arm]
    t = [S[s]["t90"] for s in sorted(S)]
    fin = [x for x in t if np.isfinite(x)]
    tm = np.mean(fin) / 1e6 if len(fin) == len(t) else np.inf
    # the pre-registered rules use the median t90: a finished seed (final_ckpt.pt) without 3 evals >= 0.9 counts as
    # inf (censored at its halt); an unfinished seed without a t90 leaves the median open (?)
    md = "    ?" if any(not np.isfinite(S[s]["t90"]) and not S[s]["done"] for s in S) else "%5.2f" % (np.median(t) / 1e6)
    per = " ".join("%.2f" % (x / 1e6) if np.isfinite(x) else (">%.1f" if S[s]["done"] else "?%.1f") % (S[s]["last_step"] / 1e6) for s, x in zip(sorted(S), t))
    de = np.mean([S[s]["drift_end"] for s in S])
    da = "/".join("%.2f" % np.nanmean([S[s]["drift_at"][i] for s in S]) if any(np.isfinite(S[s]["drift_at"][i]) for s in S)
                  else "-" for i in range(len(AT)))
    rm = "/".join("%.2f" % np.mean([S[s]["removed"][i] for s in S]) for i in range(3))
    pk = [S[s]["pick_old_head"] for s in sorted(S) if np.isfinite(S[s]["pick_old_head"])]
    pks = " ".join("%.3f" % x for x in pk) if pk else "-"
    pt = " ".join("%.3f" % S[s]["pick_t90"] if np.isfinite(S[s]["pick_t90"]) else "-" for s in sorted(S))
    print(f"{arm:10s} | {len(S):5d} | {per:22s} | {tm:5.2f} | {md:6s} | {de:9.2f} | {da:19s} | {rm:24s} | "
          + (f"{pks:26s} | {pt}" if T90 else pks))
for arm in sorted(res, key=key):
    for s in sorted(res[arm]):
        if res[arm][s]["fn_skipped"]:
            print(f"stack_{arm}_s{s}: Gram not factored in {res[arm][s]['fn_skipped']} iterations (fn.csv skipped = 1)")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1, default=float)
