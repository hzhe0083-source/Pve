"""Stage logging for the PickCube-layout tasks (PlaceCube-v1, PickCube-v1): how far each episode gets, in training
rollouts (train.csv) and in the deterministic evaluation (eval.csv), and where the policy's action means move.

Stage flags come from the env's own evaluate() (PickCubeEnv: is_grasped, is_obj_placed), read every step:
    G  grasped at some step
    H  held at the goal at some step: grasped and within goal_thresh (0.025 m) of the goal; the release decision
    S  success_once (ManiSkillVectorEnv's latch; PlaceCube: placed and not grasped)
An episode is counted in the iteration it ends in, like train_success. On a step where some env ends,
ManiSkillVectorEnv auto-resets it and returns the reset's info at the top level and the step's own info (every env)
in final_info, so the step's flags are read from final_info then (mani_skill/vector/wrappers/gymnasium.py, step()).
The reset info is never read: ManiSkill runs no physics step on reset, so its contact (grasp) flags are stale.
"""
import torch

INFO_KEYS = {"PlaceCube-v1": ("is_grasped", "is_obj_placed"), "PickCube-v1": ("is_grasped", "is_obj_placed")}
COUNTS = ("done", "grasp", "held", "held_succ", "succ", "no_release", "lost", "no_grasp")
EVAL_COLS = ("ev_grasp", "ev_held", "ev_held_succ", "ev_no_release", "ev_lost", "ev_no_grasp")
PHASE_COLS = ("mu_grip_goal", "mu_grip_carry", "mu_grip_free", "dev_arm_carry", "dev_grip_carry", "dev_arm_free",
              "dev_grip_free", "frac_goal", "frac_carry")


class EpisodeStages:
    """Per-env latches G and H over the current episode, and counts over the episodes that ended since pop():
    done, G, H, H and S, S, no_release (H, no S, still grasped at its last step), lost (G, never H, no S), no_grasp
    (never G, no S). Counts stay on the device; pop() reads them once."""

    def __init__(self, n, device, keys):
        self.kg, self.kp = keys
        self.g = torch.zeros(n, dtype=torch.bool, device=device)
        self.h = torch.zeros_like(self.g)
        self.n = torch.zeros(len(COUNTS), dtype=torch.long, device=device)

    def reset(self):
        """Every env starts a new episode (eval: eval_envs.reset())."""
        self.g.zero_()
        self.h.zero_()
        self.n.zero_()

    def update(self, infos):
        fin = "final_info" in infos
        inf = infos["final_info"] if fin else infos
        g = inf[self.kg].bool()
        self.g |= g
        self.h |= g & inf[self.kp].bool()
        if fin:
            m = infos["_final_info"].bool()
            s = inf["episode"]["success_once"].bool() & m
            G, H = self.g & m, self.h & m
            self.n += torch.stack([m.sum(), G.sum(), H.sum(), (H & s).sum(), s.sum(), (H & ~s & g).sum(),
                                   (G & ~H & ~s).sum(), (m & ~self.g & ~s).sum()])
            self.g &= ~m
            self.h &= ~m

    def pop(self):
        c = self.n.tolist()
        self.n.zero_()
        return c

    def fractions(self):
        """eval.csv: grasp, held, held_succ, no_release, lost, no_grasp as fractions of the episodes that ended."""
        c = dict(zip(COUNTS, self.pop()))
        if not c["done"]:
            return [float("nan")] * len(EVAL_COLS)
        return [c[k] / c["done"] for k in ("grasp", "held", "held_succ", "no_release", "lost", "no_grasp")]


def phase_stats(obs, valid, mu, mu0, near=0.025, far=0.05):
    """On a rollout batch (PickCube layout: grasp flag 18, cube -> goal 39:42; valid drops each episode's first
    observation, whose grasp flag is stale after an auto-reset): the mean gripper action mean (last dim) where the
    cube is held within `near` of the goal (goal), held farther than `far` (carry) and not held (free); the squared
    deviation of the action mean from the start policy's mu0 (arm dims summed, and the gripper) in carry and free; the
    share of valid states in goal and carry. NaN for an empty set."""
    held = obs[:, 18] > 0.5
    d = obs[:, 39:42].norm(dim=1)
    goal, carry, free = valid & held & (d <= near), valid & held & (d > far), valid & ~held
    dev = (mu - mu0) ** 2
    arm, grip = dev[:, :-1].sum(1), dev[:, -1]

    def mean(v, m):
        return v[m].mean().item() if bool(m.any()) else float("nan")

    nv = max(1, int(valid.sum()))
    return [mean(mu[:, -1], goal), mean(mu[:, -1], carry), mean(mu[:, -1], free), mean(arm, carry), mean(grip, carry),
            mean(arm, free), mean(grip, free), int(goal.sum()) / nv, int(carry.sum()) / nv]
