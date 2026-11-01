"""Unified object-centric state observation and a multi-task vector env over ManiSkill3 Panda tasks.

Every task is mapped onto the same slots so one policy (and one latent) can serve all of them:

    robot   qpos 9, qvel 9, tcp pose 7                                   25
    mover   pose 7, half size 3, grasped 1                               11   object that must end up somewhere
    tool    pose 7, half size 3, grasped 1, present 1                    12   object used to act on the mover
    target  pose 7, half size 3, present 1                               11   object the mover is placed on/in
    goal    position 3, present 1                                         4
    rel     tcp->mover 3, mover->goal 3                                    6
    task    one-hot over TASKS                                             9

Sizes are given as geometry (half extents) rather than object IDs so that knowledge can transfer across objects.
"""
from dataclasses import dataclass, field

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the tasks
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

TASKS = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "StackCube-v1", "PlaceSphere-v1", "PokeCube-v1",
         "LiftPegUpright-v1", "PullCubeTool-v1", "PegInsertionSide-v1"]
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
CUBE = (0.02, 0.02, 0.02)
PEG = (0.12, 0.025, 0.025)


@dataclass(frozen=True)
class TaskSpec:
    """Which observation keys and env attributes play each role in one task."""
    mover_key: str
    mover_actor: str
    mover_half: tuple | str
    tool_key: str | None = None
    tool_actor: str | None = None
    tool_half: tuple = (0.0, 0.0, 0.0)
    target_key: str | None = None          # a pose, or a position when target_is_pos
    target_is_pos: bool = False
    target_half: tuple | str = (0.0, 0.0, 0.0)
    goal_key: str | None = None            # "<key>" or "above:<pose key>" for stacking
    extra: dict = field(default_factory=dict)


SPECS = {
    "PushCube-v1": TaskSpec("obj_pose", "obj", CUBE, goal_key="goal_pos"),
    "PullCube-v1": TaskSpec("obj_pose", "obj", CUBE, goal_key="goal_pos"),
    "PickCube-v1": TaskSpec("obj_pose", "cube", CUBE, goal_key="goal_pos"),
    "StackCube-v1": TaskSpec("cubeA_pose", "cubeA", CUBE, target_key="cubeB_pose", target_half=CUBE,
                             goal_key="above:cubeB_pose"),
    "PlaceSphere-v1": TaskSpec("obj_pose", "obj", (0.02, 0.02, 0.02), target_key="bin_pos", target_is_pos=True,
                               target_half="bin", goal_key="bin_pos"),
    "PokeCube-v1": TaskSpec("cube_pose", "cube", CUBE, tool_key="peg_pose", tool_actor="peg", tool_half=PEG,
                            goal_key="goal_pos"),
    "LiftPegUpright-v1": TaskSpec("obj_pose", "peg", PEG),
    "PullCubeTool-v1": TaskSpec("cube_pose", "cube", CUBE, tool_key="tool_pose", tool_actor="l_shape_tool",
                                tool_half="l_tool"),
    "PegInsertionSide-v1": TaskSpec("peg_pose", "peg", "peg_half_size", target_key="box_hole_pose",
                                    target_half="box_hole", goal_key="box_hole_pose"),
}
OBS_DIM = 25 + 11 + 12 + 11 + 4 + 6 + len(TASKS)
# flags and the task one-hot: left unnormalised, otherwise a slot that is constant over the source tasks
# (an unseen task's one-hot, a goal flag that is always 1) becomes a huge input on a new task
TASK_DIMS = list(range(OBS_DIM - len(TASKS), OBS_DIM))
CATEGORICAL_DIMS = [35, 46, 47, 58, 62] + TASK_DIMS
GRASP_DIMS = [35, 46]  # mover grasped, tool grasped


def _const(x, n, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device).expand(n, len(x))


def _half(spec_half, extra, base, n, device):
    """Half extents of a slot object, as an (n, 3) tensor."""
    if isinstance(spec_half, tuple):
        return _const(spec_half, n, device)
    if spec_half == "peg_half_size":
        return extra["peg_half_size"].float()
    if spec_half == "box_hole":
        r = extra["box_hole_radius"].float().reshape(n, 1)
        return torch.cat([torch.full_like(r, 0.0), r, r], dim=1)
    if spec_half == "bin":
        return _const((base.block_half_size[1], base.block_half_size[1], base.edge_block_half_size[2]), n, device)
    if spec_half == "l_tool":
        return _const((getattr(base, "handle_length", 0.2) / 2, getattr(base, "hook_length", 0.05) / 2,
                       getattr(base, "width", 0.05) / 2), n, device)
    raise KeyError(spec_half)


def _grasped(base, actor_name, n, device):
    if actor_name is None:
        return torch.zeros(n, 1, device=device)
    return base.agent.is_grasping(getattr(base, actor_name)).float().reshape(n, 1)


def unify(task, obs, base):
    """Map one task's state_dict observation onto the shared slot layout, returning an (n, OBS_DIM) tensor."""
    spec = SPECS[task]
    agent, extra = obs["agent"], obs["extra"]
    n, device = agent["qpos"].shape[0], agent["qpos"].device
    zeros = lambda k: torch.zeros(n, k, device=device)  # noqa: E731
    ones = lambda: torch.ones(n, 1, device=device)  # noqa: E731

    robot = torch.cat([agent["qpos"], agent["qvel"], extra["tcp_pose"]], dim=1)
    mover_pose = extra[spec.mover_key]
    mover = torch.cat([mover_pose, _half(spec.mover_half, extra, base, n, device),
                       _grasped(base, spec.mover_actor, n, device)], dim=1)
    if spec.tool_key:
        tool = torch.cat([extra[spec.tool_key], _half(spec.tool_half, extra, base, n, device),
                          _grasped(base, spec.tool_actor, n, device), ones()], dim=1)
    else:
        tool = zeros(12)
    if spec.target_key:
        t = extra[spec.target_key]
        t_pose = torch.cat([t, _const(IDENTITY_QUAT, n, device)], dim=1) if spec.target_is_pos else t
        target = torch.cat([t_pose, _half(spec.target_half, extra, base, n, device), ones()], dim=1)
    else:
        target = zeros(11)
    if spec.goal_key is None:
        goal_pos, goal = None, zeros(4)
    else:
        if spec.goal_key.startswith("above:"):
            below = extra[spec.goal_key.split(":", 1)[1]]
            goal_pos = below[:, :3] + torch.tensor([0.0, 0.0, 2 * CUBE[2]], device=device)
        else:
            goal_pos = extra[spec.goal_key][:, :3]
        goal = torch.cat([goal_pos, ones()], dim=1)
    tcp_to_mover = mover_pose[:, :3] - extra["tcp_pose"][:, :3]
    mover_to_goal = goal_pos - mover_pose[:, :3] if goal_pos is not None else zeros(3)
    task_onehot = zeros(len(TASKS))
    task_onehot[:, TASKS.index(task)] = 1.0
    out = torch.cat([robot, mover, tool, target, goal, tcp_to_mover, mover_to_goal, task_onehot], dim=1)
    assert out.shape[1] == OBS_DIM, (task, out.shape)
    return out


class MultiTaskEnv:
    """Several ManiSkill GPU-sim tasks in one process, stepped together with one unified observation tensor.

    Rows are grouped by task in the order of `tasks`; `task_ids` gives each row's index into TASKS.
    """

    def __init__(self, tasks, num_envs_per_task, *, reward_mode="normalized_dense",
                 control_mode="pd_joint_delta_pos", seed=0):
        self.tasks = list(tasks)
        self.n = num_envs_per_task
        self.envs = []
        for t in self.tasks:
            e = gym.make(t, num_envs=self.n, obs_mode="state_dict", control_mode=control_mode,
                         reward_mode=reward_mode, sim_backend="physx_cuda", render_mode=None)
            self.envs.append(ManiSkillVectorEnv(e, self.n, ignore_terminations=False, record_metrics=True))
        self.device = self.envs[0].unwrapped.device
        self.task_ids = torch.cat([torch.full((self.n,), TASKS.index(t), device=self.device) for t in self.tasks])
        self.num_envs = self.n * len(self.tasks)
        self.act_dim = self.envs[0].single_action_space.shape[0]
        self.seed = seed

    def _unify_all(self, obs_list):
        return torch.cat([unify(t, o, e.unwrapped) for t, o, e in zip(self.tasks, obs_list, self.envs)], dim=0)

    def reset(self):
        obs = [e.reset(seed=self.seed + i)[0] for i, e in enumerate(self.envs)]
        return self._unify_all(obs)

    def step(self, actions):
        """actions: (num_envs, act_dim). Returns obs, reward, terminated, truncated, success, final_obs or None."""
        obs, rew, term, trunc, succ, final = [], [], [], [], [], []
        any_final = False
        for i, (t, e) in enumerate(zip(self.tasks, self.envs)):
            o, r, te, tr, info = e.step(actions[i * self.n:(i + 1) * self.n])
            obs.append(o)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
            succ.append(info["success"].bool())
            if "final_observation" in info:
                any_final = True
                final.append(unify(t, info["final_observation"], e.unwrapped))
            else:
                final.append(None)
        final_obs = None
        if any_final:
            cur = self._unify_all(obs)
            final_obs = torch.cat([f if f is not None else cur[i * self.n:(i + 1) * self.n]
                                   for i, f in enumerate(final)], dim=0)
        return (self._unify_all(obs), torch.cat(rew), torch.cat(term), torch.cat(trunc), torch.cat(succ), final_obs)

    def close(self):
        for e in self.envs:
            e.close()
