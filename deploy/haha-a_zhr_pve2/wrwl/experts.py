"""Load single-task experts trained with ManiSkill's official ppo.py and act with them."""
import os
import sys
import types
from pathlib import Path

import torch
from mani_skill.utils import common

ROOT = Path(os.environ.get("PVE_ROOT", Path.home() / "pve"))   # another checkout on a shared lab machine
PPO_DIR = ROOT / "third_party/ms_ppo"
EXPERT_DIR = ROOT / "runs/experts/runs"
sys.path.insert(0, str(PPO_DIR))
from ppo import Agent  # noqa: E402  the official 3x256 Tanh actor-critic

EXPERT_NAMES = {
    "PushCube-v1": "PushCube", "PullCube-v1": "PullCube", "PickCube-v1": "PickCube", "StackCube-v1": "StackCube",
    "PlaceSphere-v1": "PlaceSphere", "PokeCube-v1": "PokeCube", "LiftPegUpright-v1": "LiftPegUpright",
    "PullCubeTool-v1": "PullCubeTool",
}


def flat_state(obs):
    """The flat 'state' observation ppo.py trained on, rebuilt from a state_dict observation."""
    return common.flatten_state_dict(obs, use_torch=True)


def load_expert(task, obs_dim, act_dim=8, ckpt="final_ckpt.pt", device="cuda"):
    spaces = types.SimpleNamespace(single_observation_space=types.SimpleNamespace(shape=(obs_dim,)),
                                   single_action_space=types.SimpleNamespace(shape=(act_dim,)))
    agent = Agent(spaces).to(device)
    agent.load_state_dict(torch.load(EXPERT_DIR / EXPERT_NAMES[task] / ckpt, map_location=device))
    agent.eval()
    return agent


def load_policy(path, obs_dim, act_dim=8, device="cuda"):
    """Any checkpoint with the official actor-critic layout (official ppo.py or wrwl.ppo flat-state runs)."""
    spaces = types.SimpleNamespace(single_observation_space=types.SimpleNamespace(shape=(obs_dim,)),
                                   single_action_space=types.SimpleNamespace(shape=(act_dim,)))
    agent = Agent(spaces).to(device)
    agent.load_state_dict(torch.load(path, map_location=device))
    agent.eval()
    return agent
