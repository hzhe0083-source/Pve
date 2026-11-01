"""Synthetic run directories for scripts/why_test.py: each scenario fixes t90 and p(held) per arm and seed, and the
script's verdicts must come out as the scenario was built.
  python synth_why.py <out_dir> <scenario>   (gating | nogating | norep | edge)"""
import json
import sys
from pathlib import Path

I, B = 102_400, 51_200
REF = {"nsa64": [6, 7, 6, 6, 6, 6, 7, 6], "npr": [10, 9, 9, 9, 8, 8, 11, 9]}     # protect_np, in intervals
ARMS = {"nsa64": (True, None, "heads", 0.0, None, 0.0), "npr": (False, "rand", "heads", 0.0, None, 0.0),
        "rp": (True, None, "random", 0.0, None, 0.0), "rf01": (False, "rand", "heads", 0.1, None, 0.0),
        "rf1": (False, "rand", "heads", 1.0, None, 0.0), "rf10": (False, "rand", "heads", 10.0, None, 0.0),
        "nsa64d": (True, None, "heads", 0.0, None, 0.05), "nprd": (False, "rand", "heads", 0.0, None, 0.05)}
ARMS = {k: v + ("null",) for k, v in ARMS.items()}
ARMS.update({"vis": (True, None, "heads", 0.0, None, 0.0, "visible"), "none": (True, None, "heads", 0.0, None, 0.0, "zero"),
             "nsa64z": (True, "zero", "heads", 0.0, None, 0.0, "null")})
for base in ("nsa64", "npr"):
    for tag, lr in {"lo": 1e-4, "hi": 1e-3, "vlo": 3e-5, "vhi": 3e-3}.items():
        ARMS[base + tag] = ARMS[base][:4] + (lr,) + ARMS[base][5:]


def write(root, arm, seed, t90_iv, p, fail_mix=(0.5, 0.3, 0.2), md5="abc"):
    """t90_iv in intervals; p = share of training episodes held at the goal (constant); fail_mix = (no_release, lost,
    no_grasp) shares of the eval failures."""
    run = root / f"place_{arm}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    proj, init, how, rest, alr, cost, mask = ARMS[arm]
    ek = {"robot_uids": "panda_wristcam", "reward_mode": "sparse"}
    if cost:
        ek["stage1_cost"] = cost
    json.dump({"exp_name": run.name, "seed": seed, "env_id": "PlaceCube-v1", "env_kwargs": json.dumps(ek),
               "halt_at": 8_000_000, "stop_success": 0.9, "stop_patience": 3, "num_envs": 1024, "num_steps": 50,
               "eval_freq": 2, "num_eval_envs": 256, "learning_rate": 3e-4, "gamma": 0.95, "min_grip_std": 1.0,
               "nullspace_adapter": 64, "nsa_project": proj, "nsa_init": init, "nsa_proj": how, "nsa_restore": rest,
               "adapter_lr": alr, "head_lr": None, "nsa_freeze_adapter": False, "nsa_mask": mask, "code_md5": md5},
              open(run / "args.json", "w"))
    n_ev = t90_iv + 3
    with open(run / "eval.csv", "w") as f:
        f.write("iteration,global_step,success_once,success_at_end,return,prior_coef,wall_s,ev_grasp,ev_held,"
                "ev_held_succ,ev_no_release,ev_lost,ev_no_grasp\n")
        for k in range(n_ev):
            s = 0.95 if k >= t90_iv else min(0.85, 0.1 * k)
            fl = 1 - s
            f.write(f"{2 * k + 1},{k * I},{s},0,0,0,0,1,0.9,{s},{fl * fail_mix[0]},{fl * fail_mix[1]},"
                    f"{fl * fail_mix[2]}\n")
    with open(run / "train.csv", "w") as f:
        f.write("iteration,global_step,train_success,train_episodes,st_done,st_grasp,st_held,st_held_succ,st_succ,"
                "st_no_release,st_lost,st_no_grasp,mu_grip_goal,mu_grip_carry,dev_arm_carry,dev_grip_carry,kl_per_decision\n")
        for it in range(1, 2 * (t90_iv + 2) + 1):
            n = 2000
            held = round(p * n)
            succ = round(0.97 * held)
            dev = 0.02 * it * (3 if p < 0.9 else 1)       # carry-state drift: 3x in a world where p dropped
            f.write(f"{it},{it * B},{succ / n},{n},{n},{n},{held},{succ},{succ},0,{n - held},0,{-1 + 0.1 * it},-1,"
                    f"{dev},{dev},{0.02 * (2 if p < 0.9 else 1)}\n")
    (run / "final_ckpt.pt").write_bytes(b"")


def scenario(root, name):
    S = range(1, 9)
    if name == "gating":    # npr loses stage-1: p 0.6 vs 0.93; everything else follows gating
        for s in S:
            n, r = REF["nsa64"][s - 1], REF["npr"][s - 1]
            write(root, "nsa64", s, n, 0.93)
            write(root, "npr", s, r, 0.93 * n / r, fail_mix=(0.1, 0.6, 0.3))
            write(root, "rp", s, r, 0.93 * n / r)
            write(root, "rf01", s, r, 0.93 * n / r)
            write(root, "rf1", s, r - 1, 0.93 * n / (r - 1))
            write(root, "rf10", s, n, 0.92)
            write(root, "nsa64d", s, n, 0.95)
            write(root, "nprd", s, n + (s % 2), 0.94)
            write(root, "nsa64lo", s, n + 3, 0.94)
            write(root, "nprlo", s, n + 5, 0.93 * (n + 3) / (n + 5))
            write(root, "nsa64hi", s, n + 1, 0.92)
            write(root, "nprhi", s, r + 3, 0.92 * (n + 1) / (r + 3))
    elif name in ("nogating", "edge"):      # p equal, slowdown not via p; lr tuning closes it (nogating)
        for s in S:
            n, r = REF["nsa64"][s - 1], REF["npr"][s - 1]
            write(root, "nsa64", s, n, 0.93)
            write(root, "npr", s, r, 0.93, fail_mix=(0.8, 0.1, 0.1))
            write(root, "rp", s, r + (1 if s % 3 == 0 else 0), 0.93)
            for x in ("rf01", "rf1", "rf10"):
                write(root, x, s, r, 0.93)
            write(root, "nsa64d", s, n, 0.93)
            write(root, "nprd", s, r, 0.93)
            write(root, "nsa64lo", s, n + 2, 0.93)
            write(root, "nsa64hi", s, n + 1, 0.93)
            if name == "nogating":
                write(root, "nprlo", s, n, 0.93)        # tuned npr matches nsa64
                write(root, "nprhi", s, r + 2, 0.93)
                write(root, "nprvlo", s, n + 2, 0.93)
            else:
                write(root, "nprlo", s, r - 1, 0.93)    # best at the low edge, no vlo run yet
                write(root, "nprhi", s, r + 2, 0.93)
    elif name in ("routing", "additive"):    # rule F: the visible channel alone is the prefix sum (routing) or adds
        for s in S:
            n, r = REF["nsa64"][s - 1], REF["npr"][s - 1]
            write(root, "nsa64", s, n, 0.93)
            write(root, "npr", s, r, 0.93)
            write(root, "none", s, n + 6, 0.93)                                 # head-only-like: 12 intervals
            write(root, "vis", s, r if name == "routing" else n + 6 + (r - n), 0.93)
            write(root, "nsa64z", s, n + (1 if s % 4 == 0 else 0) if name == "routing" else n + 3, 0.93)
    elif name == "norep":
        for s in S:
            write(root, "nsa64", s, REF["nsa64"][s - 1] + 4, 0.93)
            write(root, "npr", s, REF["npr"][s - 1], 0.93)
    # Rule 0 inputs: protect_np-like runs (train_success only)
    np_root = root.parent / "np_runs"
    for s in S:
        for arm, src, ts in (("nsa64", "nsa64", 0.95), ("nsa64npr", "npr", 0.62 if name == "gating" else 0.93)):
            run = np_root / f"place_{arm}_s{s}"
            run.mkdir(parents=True, exist_ok=True)
            t = REF[src][s - 1]
            json.dump({"exp_name": run.name}, open(run / "args.json", "w"))
            with open(run / "eval.csv", "w") as f:
                f.write("iteration,global_step,success_once\n")
                for k in range(t + 3):
                    f.write(f"{2 * k + 1},{k * I},{0.95 if k >= t else 0.1}\n")
            with open(run / "train.csv", "w") as f:
                f.write("iteration,global_step,train_success,train_episodes\n")
                for it in range(1, 2 * (t + 2) + 1):
                    f.write(f"{it},{it * B},{ts},2000\n")
            (run / "final_ckpt.pt").write_bytes(b"")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    scenario(out / "runs", sys.argv[2])
