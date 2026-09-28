# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/custom_rlbench_env.py
from __future__ import annotations

import os
from pathlib import Path

from bridgevla.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv

from utils.eval_pack import load_demo, read_manifest
from utils.improved_eval import force_renderable_ghosts


def _resolve_eval_pack_root(dataset_root: str, explicit: str | None) -> Path | None:
    if explicit:
        root = Path(explicit)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"eval pack manifest not found: {root}")
        return root
    env_root = os.environ.get("BRIDGEVLA_EVAL_PACK", "").strip()
    if env_root:
        root = Path(env_root)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"BRIDGEVLA_EVAL_PACK manifest not found: {root}")
        return root
    if dataset_root:
        root = Path(dataset_root)
        manifest = root / "manifest.json"
        if manifest.is_file():
            data = read_manifest(root)
            if data.get("format") == "bridgevla_eval_pack":
                return root
    return None


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, eval_pack_root=None, **kwargs):
        # Capture before super(); YARR parent does not keep dataset_root on self.
        dataset_root = kwargs.get("dataset_root", "")
        if not dataset_root and len(args) >= 4:
            # task_classes, observation_config, action_mode, dataset_root, ...
            dataset_root = args[3]
        self._dataset_root = dataset_root
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)
        self._eval_pack_root = _resolve_eval_pack_root(
            self._dataset_root, eval_pack_root
        )

    def reset(self) -> dict:
        super().reset()
        if force_renderable_ghosts(self._task) > 0:
            self._previous_obs_dict = self.extract_obs(self._task.get_observation())
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        if self._eval_pack_root is not None:
            # Pre-encoded eval pack: no expanded tree, no image IO.
            task_name = self._task.get_name()
            d = load_demo(self._eval_pack_root, task_name, int(i))
        else:
            d = self._task.get_demos(
                1, live_demos=False, random_selection=False, from_episode_number=i
            )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]
        if force_renderable_ghosts(self._task) > 0:
            obs = self._task.get_observation()

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
