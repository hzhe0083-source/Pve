"""PickYCBSubset-v1: PickSingleYCB-v1 on a fixed subset of YCB models, with the arm starting where the PickCube expert
starts (TableSceneBuilder's panda_wristcam pose, wrist at -pi/4) instead of PickSingleYCB's own higher pose with the
wrist at +pi/4 (pick_single_ycb.py:186-197). Object and goal sampling (pick_single_ycb.py:171-184, the same draws in
the same order), success (object within 2.5 cm of the goal, robot static; no release) and rewards are PickSingleYCB's.
The observation is PickSingleYCB's (45-dim, with the grasp flag); adapters.ycb_to_pick reads it in PickCube's layout.
With several envs each keeps one model for the whole run (reconfiguration_freq 0); envs built with
reconfiguration_freq=1, as ppo.py's evaluation envs, draw new models at every reset. ycb_ids keeps the full list.
Needs the YCB assets: python -m mani_skill.utils.download_asset ycb -y.

The subset is fixed by a geometric rule, written down before any suite run and never by a success rate (the
zero-shot gate then keeps or drops the task as a whole; the list is not edited): from PickSingleYCB's model list
(pick_single_ycb.py:65-79), every model whose bounding box in assets/mani_skill2_ycb/info_pick_v0.json, times the
model's scale as actors/ycb.py builds it (the object stands in that frame, pick_single_ycb.py:168, 176-178), has
    every extent in [RULE_MIN_EXTENT, RULE_MAX_EXTENT] = [0.02, 0.12] m (compact: no plates, cutlery, long tools or
        big boxes), and
    the smaller horizontal extent <= RULE_MAX_SPAN = 0.07 m (the open Panda gripper, 0.08 m, spans it from above).
model_ids="rule" (the default) applies it; a list keeps exactly those models; None keeps every model.
scripts/smoke_envs.py --tasks=ycb prints the list the rule gives, with each model's extents.
"""
import numpy as np
import torch
from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

RULE_MIN_EXTENT, RULE_MAX_EXTENT, RULE_MAX_SPAN = 0.02, 0.12, 0.07
RULE_MIN_MODELS = 6         # fewer: the rule does not fit this asset version, and the task counts as unavailable


def model_extents(info, model_id):
    """(x, y, z) extents in metres of model_id's bounding box at the scale the builder uses (actors/ycb.py)."""
    meta = info[model_id]
    scale = meta.get("scales", [1.0])[0]
    return (np.asarray(meta["bbox"]["max"], dtype=float) - np.asarray(meta["bbox"]["min"], dtype=float)) * scale


def rule_models(model_ids, info):
    """The models of model_ids that the geometric rule keeps, in the given order."""
    keep = []
    for m in model_ids:
        e = model_extents(info, m)
        if e.min() >= RULE_MIN_EXTENT and e.max() <= RULE_MAX_EXTENT and min(e[0], e[1]) <= RULE_MAX_SPAN:
            keep.append(m)
    return keep


@register_env("PickYCBSubset-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickYCBSubsetEnv(PickSingleYCBEnv):
    def __init__(self, *args, model_ids="rule", **kwargs):
        self.model_subset = model_ids if model_ids is None or model_ids == "rule" else list(model_ids)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        if getattr(self, "ycb_ids", None) is None:      # the first call sees PickSingleYCB's full list
            self.ycb_ids = self.all_model_ids
            if self.model_subset == "rule":
                info = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
                self.model_subset = rule_models(self.ycb_ids.tolist(), info)
                assert len(self.model_subset) >= RULE_MIN_MODELS, \
                    f"the YCB rule keeps {self.model_subset}, fewer than {RULE_MIN_MODELS} models"
            elif self.model_subset is not None:
                missing = sorted(set(self.model_subset) - set(self.ycb_ids.tolist()))
                assert not missing, f"not PickSingleYCB models: {missing}; the models are {sorted(self.ycb_ids.tolist())}"
        if self.model_subset is not None:
            self.all_model_ids = np.array(self.model_subset)
        super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)    # table, and the arm at the panda_wristcam start (Pick's)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
