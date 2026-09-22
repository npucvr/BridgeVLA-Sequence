#!/usr/bin/env python3
# gbw____
"""Independent RLBench evaluator for Original BridgeVLA model_80.

This file deliberately does not replace eval.py.  It keeps the Original
discrete rotation head and checkpoint, while applying only the evaluator-side
changes audited from the BridgeVLA++ reference:

* fixed dataset episodes 0..24;
* task-specific action budgets;
* correct routing of the 9-D action collision flag;
* velocity-based arm settling around gripper execution;
* renderable ghost repair and step-0 re-capture;
* runtime and per-episode audit records.

Output contract:
* ``episode_results.jsonl`` is appended and flushed after every episode;
* ``eval_results.csv`` writes its header before launch and appends one row after
  every completed task;
* ``progress.json`` is atomically replaced after every episode/task and marks
  an interrupted run without creating a false final ``summary.json``.
"""

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
import yaml

from pyrep.objects.shape import Shape
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.utils.rollout_generator import RolloutGenerator

import bridgevla
import pyrep
import rlbench
from bridgevla.libs.peract.helpers import utils
from bridgevla.utils.rvt_utils import RLBENCH_TASKS
from utils.peract_utils_rlbench import CAMERAS, IMAGE_SIZE, SCENE_BOUNDS
from eval import load_agent
from bridgevla.libs.peract.helpers.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv as BaseCustomMultiTaskRLBenchEnv,
)


# The list is copied from the B++ evaluator-side asset repair.  These are
# renderable flags only; no physics, task success criterion, or model input is
# modified.
GHOST_RENDERABLE_FIX = {
    "put_item_in_drawer": ["item"],
    "put_money_in_safe": ["dollar_front_visual", "dollar_back_visual"],
    "light_bulb_in": ["lamp_base", "lamp_screw"],
    "place_cups": ["place_cups_holder_spoke3"],
    "place_shape_in_shape_sorter": ["shape_sorter_top"],
}


class ImprovedCustomMultiTaskRLBenchEnv(BaseCustomMultiTaskRLBenchEnv):
    """Current workspace environment plus the B++ renderable-asset repair."""

    def _force_renderable_ghosts(self):
        if os.environ.get("IMPROVED_ASSET_FIX", "1") != "1":
            return 0
        try:
            task_name = self._task._task.get_name()
        except Exception:
            return 0
        count = 0
        for object_name in GHOST_RENDERABLE_FIX.get(task_name, []):
            try:
                Shape(object_name).set_renderable(True)
                count += 1
            except Exception:
                # A task variation can omit an optional visual object.
                continue
        return count

    def reset(self):
        observation = super().reset()
        if self._force_renderable_ghosts() > 0:
            self._previous_obs_dict = self.extract_obs(
                self._task.get_observation()
            )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1
        self._i = 0
        self._task.set_variation(-1)
        demo = self._task.get_demos(
            1,
            live_demos=False,
            random_selection=False,
            from_episode_number=i,
        )[0]
        self._task.set_variation(demo.variation_number)
        description, observation = self._task.reset_to_demo(demo)
        self._lang_goal = description[0]
        if self._force_renderable_ghosts() > 0:
            observation = self._task.get_observation()
        self._previous_obs_dict = self.extract_obs(observation)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict


class ImprovedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
    """B++ workspace clipping retained for the Original action head."""

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions=True):
        action[:3] = np.clip(
            action[:3],
            np.asarray(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.asarray(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )
        return super().action(scene, action, ignore_collisions)


class ImprovedMoveArmThenGripper(MoveArmThenGripper):
    """Route [pose(7), gripper(1), collision(1)] without changing the policy."""

    def _wait_arm_stopped(self, scene: Scene):
        max_velocity = float(os.environ.get("IMPROVED_SETTLE_MAX_VEL", "0.01"))
        max_steps = int(os.environ.get("IMPROVED_SETTLE_MAX_STEPS", "100"))
        if os.environ.get("IMPROVED_SETTLE", "1") != "1":
            return
        for _ in range(max_steps):
            velocity = np.asarray(scene.robot.arm.get_joint_velocities())
            if velocity.size == 0 or np.max(np.abs(velocity)) < max_velocity:
                break
            scene.step()

    def action(self, scene: Scene, action: np.ndarray):
        arm_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
        gripper_size = int(np.prod(self.gripper_action_mode.action_shape(scene)))
        action = np.asarray(action)
        arm_action = np.asarray(action[:arm_size])
        gripper_action = np.asarray(
            action[arm_size : arm_size + gripper_size]
        )
        collision_tail = action[arm_size + gripper_size :]
        if collision_tail.size:
            ignore_collisions = bool(int(round(float(collision_tail[0]))))
        else:
            ignore_collisions = True
        observations = self.arm_action_mode.action(
            scene, arm_action, ignore_collisions
        )
        self._wait_arm_stopped(scene)
        self.gripper_action_mode.action(scene, gripper_action)
        self._wait_arm_stopped(scene)
        return observations


def _module_commit(module):
    """Best-effort commit audit without requiring a nested git checkout."""
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return ""
    candidate = Path(module_path).resolve()
    for parent in (candidate.parent, *candidate.parents):
        if not (parent / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return ""


def _runtime_metadata(
    step_limits, tasks, model_path, dataset_root, output_dir, episode_count
):
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": os.environ.get("DEVICE", "0"),
        "project_root": os.environ.get("PROJECT_ROOT", ""),
        "dataset_root": os.path.abspath(dataset_root),
        "model_path": os.path.abspath(model_path),
        "output_dir": os.path.abspath(output_dir),
        "coppeliasim_root": os.environ.get("COPPELIASIM_ROOT", ""),
        "display": os.environ.get("DISPLAY", ""),
        "rlbench_file": getattr(rlbench, "__file__", ""),
        "pyrep_file": getattr(pyrep, "__file__", ""),
        "bridgevla_file": getattr(bridgevla, "__file__", ""),
        "rlbench_commit": _module_commit(rlbench),
        "pyrep_commit": _module_commit(pyrep),
        "step_limits": step_limits,
        "tasks": tasks,
        "episode_ids": list(range(episode_count)),
        "episode_selection": "fixed contiguous dataset episodes from episode0",
        "action_wrapper": "ImprovedMoveArmThenGripper",
        "rotation_head": "Original BridgeVLA discrete rot_ver=1",
        "asset_fix_enabled": os.environ.get("IMPROVED_ASSET_FIX", "1") == "1",
        "settle_enabled": os.environ.get("IMPROVED_SETTLE", "1") == "1",
        "settle_max_velocity": float(
            os.environ.get("IMPROVED_SETTLE_MAX_VEL", "0.01")
        ),
        "settle_max_steps": int(
            os.environ.get("IMPROVED_SETTLE_MAX_STEPS", "100")
        ),
    }


def _check_fixed_dataset(dataset_root, tasks, episode_count):
    for task_name in tasks:
        episode_root = Path(dataset_root) / task_name / "all_variations" / "episodes"
        if not episode_root.is_dir():
            raise FileNotFoundError("missing task episode directory: " + str(episode_root))
        missing = [
            str(episode_root / ("episode" + str(index)))
            for index in range(episode_count)
            if not (episode_root / ("episode" + str(index))).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "fixed episode plan requires episode0..episode"
                + str(episode_count - 1)
                + "; missing "
                + ", ".join(missing[:3])
            )


RESULT_FIELDS = ["task", "success_rate", "successes", "episodes", "step_limit"]


def _flush_handle(handle, sync=False):
    """Make an append visible immediately, optionally forcing it to disk."""
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def _write_progress(
    output_dir,
    tasks,
    task_results,
    completed_episodes,
    episode_count,
    status,
    episode_records=None,
):
    """Atomically publish resumable progress for an in-flight repeat."""
    success_records = (
        episode_records if episode_records is not None else task_results
    )
    successes = sum(
        int(item.get("successes", item.get("success", False)))
        for item in success_records
    )
    expected_episodes = len(tasks) * episode_count
    payload = {
        "status": status,
        "completed_tasks": len(task_results),
        "expected_tasks": len(tasks),
        "completed_episodes": completed_episodes,
        "expected_episodes": expected_episodes,
        "successes": successes,
        "success_rate_so_far": (
            successes / float(completed_episodes)
            if completed_episodes
            else 0.0
        ),
        "task_results": task_results,
    }
    temporary = Path(output_dir) / ("progress.json.tmp." + str(os.getpid()))
    final = Path(output_dir) / "progress.json"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        _flush_handle(handle, sync=True)
    os.replace(temporary, final)


def _append_task_result(writer, handle, task_result):
    """Append exactly one completed-task row without changing the legacy CSV schema."""
    writer.writerow(task_result)
    _flush_handle(handle, sync=True)


@torch.no_grad()
def run_evaluation(
    agent,
    tasks,
    dataset_root,
    output_dir,
    step_limits,
    episode_count=25,
    start_episode=0,
    device=0,
    model_path="",
):
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    observation_config = utils.create_obs_config(
        CAMERAS, camera_resolution, method_name=""
    )
    action_mode = ImprovedMoveArmThenGripper(
        ImprovedEndEffectorPoseViaPlanning(),
        Discrete(),
    )
    task_files = {
        name[:-3]
        for name in os.listdir(rlbench_task.TASKS_PATH)
        if name.endswith(".py") and name != "__init__.py"
    }
    unknown = [task for task in tasks if task not in task_files]
    if unknown:
        raise ValueError("unknown RLBench tasks: " + ", ".join(unknown))
    task_classes = [task_file_to_task_class(task) for task in tasks]
    environment = ImprovedCustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=observation_config,
        action_mode=action_mode,
        dataset_root=dataset_root,
        episode_length=25,
        headless=True,
        swap_task_every=episode_count,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=-1,
    )
    environment.eval = True
    rollout_generator = RolloutGenerator("cuda:" + str(device))
    step_signal = Value("i", -1)
    episode_records = []
    task_results = []
    started = time.time()
    os.makedirs(output_dir, exist_ok=True)
    episode_path = Path(output_dir) / "episode_results.jsonl"
    csv_path = Path(output_dir) / "eval_results.csv"
    episode_handle = open(episode_path, "w", encoding="utf-8")
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_handle, fieldnames=RESULT_FIELDS)
    csv_writer.writeheader()
    _flush_handle(csv_handle, sync=True)
    _write_progress(
        output_dir,
        tasks,
        task_results,
        completed_episodes=0,
        episode_count=episode_count,
        status="running",
        episode_records=episode_records,
    )
    environment_launched = False
    completed = False
    try:
        environment.launch()
        environment_launched = True
        for task_index, task_name in enumerate(tasks):
            if task_index > 0 and hasattr(agent, "end_episode"):
                agent.end_episode("task_switch")
            limit = int(step_limits[task_name])
            environment._episode_length = limit
            task_records = []
            for episode_offset in range(episode_count):
                dataset_episode = start_episode + episode_offset
                record = {
                    "task": task_name,
                    "episode": dataset_episode,
                    "step_limit": limit,
                    "status": "error",
                    "success": False,
                    "steps": 0,
                    "reward": 0.0,
                    "error": "",
                }
                rollout = []
                try:
                    generator = rollout_generator.generator(
                        step_signal=step_signal,
                        env=environment,
                        agent=agent,
                        episode_length=limit,
                        timesteps=1,
                        eval=True,
                        eval_demo_seed=dataset_episode,
                        record_enabled=False,
                        replay_ground_truth=False,
                    )
                    rollout = list(generator)
                    if rollout:
                        final_transition = rollout[-1]
                        record["reward"] = float(final_transition.reward)
                        record["success"] = bool(record["reward"] > 99.0)
                        record["status"] = "success" if record["success"] else "failure"
                        record["steps"] = len(rollout)
                        if (
                            final_transition.terminal
                            or getattr(final_transition, "timeout", False)
                        ) and hasattr(agent, "end_episode"):
                            reason = (
                                "timeout"
                                if getattr(final_transition, "timeout", False)
                                else "terminal"
                            )
                            agent.end_episode(reason)
                    else:
                        record["error"] = "empty rollout"
                except Exception as error:
                    record["error"] = repr(error)
                    if hasattr(agent, "end_episode"):
                        agent.end_episode("episode_error")
                task_records.append(record)
                episode_records.append(record)
                episode_handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                _flush_handle(episode_handle)
                _write_progress(
                    output_dir,
                    tasks,
                    task_results,
                    completed_episodes=len(episode_records),
                    episode_count=episode_count,
                    status="running",
                    episode_records=episode_records,
                )
                print(
                    "[episode] task="
                    + task_name
                    + " episode="
                    + str(dataset_episode)
                    + " status="
                    + record["status"]
                    + " steps="
                    + str(record["steps"]),
                    flush=True,
                )
            successes = sum(int(item["success"]) for item in task_records)
            task_results.append(
                {
                    "task": task_name,
                    "success_rate": successes / float(episode_count),
                    "successes": successes,
                    "episodes": episode_count,
                    "step_limit": limit,
                }
            )
            _append_task_result(csv_writer, csv_handle, task_results[-1])
            _write_progress(
                output_dir,
                tasks,
                task_results,
                completed_episodes=len(episode_records),
                episode_count=episode_count,
                status="running",
                episode_records=episode_records,
            )
            print(
                "[task] "
                + task_name
                + " "
                + str(successes)
                + "/"
                + str(episode_count)
                + " ("
                + format(100.0 * successes / episode_count, ".2f")
                + "%)",
                flush=True,
            )
        completed = True
    finally:
        try:
            if environment_launched:
                environment.shutdown()
        finally:
            _flush_handle(episode_handle, sync=True)
            _flush_handle(csv_handle, sync=True)
            episode_handle.close()
            csv_handle.close()
            if not completed:
                try:
                    _write_progress(
                        output_dir,
                        tasks,
                        task_results,
                        completed_episodes=len(episode_records),
                        episode_count=episode_count,
                        status="interrupted_or_failed",
                        episode_records=episode_records,
                    )
                except Exception:
                    # Preserve the original evaluation exception if progress
                    # publication itself fails during cleanup.
                    pass

    total_successes = sum(item["successes"] for item in task_results)
    total_episodes = sum(item["episodes"] for item in task_results)
    summary = {
        "method": "A0-improved-evaluator",
        "filter_mode": "none",
        "tasks": tasks,
        "episodes_per_task": episode_count,
        "start_episode": start_episode,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / float(total_episodes),
        "task_results": task_results,
        "runtime_seconds": time.time() - started,
        "runtime": _runtime_metadata(
            step_limits,
            tasks,
            model_path,
            dataset_root,
            output_dir,
            episode_count,
        ),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_progress(
        output_dir,
        tasks,
        task_results,
        completed_episodes=len(episode_records),
        episode_count=episode_count,
        status="completed",
        episode_records=episode_records,
    )
    print(
        "[complete] "
        + str(total_successes)
        + "/"
        + str(total_episodes)
        + " = "
        + format(100.0 * total_successes / total_episodes, ".2f")
        + "%",
        flush=True,
    )
    return summary


def main():
    project_root = os.environ.get(
        "PROJECT_ROOT", "/remote_userdata/gaobowen/BridgeVLA-Sequence"
    )
    dataset_root = os.environ.get(
        "DATASET_ROOT", os.path.join(project_root, "data/datasets/rl_eval")
    )
    model_path = os.environ.get(
        "MODEL_PATH", os.path.join(project_root, "data/ckpt/rl/model_80.pth")
    )
    exp_cfg_path = os.environ.get(
        "EXP_CFG_PATH", os.path.join(project_root, "data/ckpt/rl/exp_cfg.yaml")
    )
    mvt_cfg_path = os.environ.get(
        "MVT_CFG_PATH", os.path.join(project_root, "data/ckpt/rl/mvt_cfg.yaml")
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR", os.path.join(project_root, "outputs/improved_eval")
    )
    step_config = os.environ.get(
        "STEP_LIMIT_CONFIG",
        os.path.join(
            project_root, "finetune/RLBench/configs/improved_eval_step_limit.yml"
        ),
    )
    task_text = os.environ.get("TASKS", "all").replace(",", " ").split()
    tasks = list(RLBENCH_TASKS) if task_text == ["all"] else task_text
    episode_count = int(os.environ.get("EVAL_EPISODES", "25"))
    start_episode = int(os.environ.get("START_EPISODE", "0"))
    device = int(os.environ.get("DEVICE", "0"))
    smoke_mode = os.environ.get("IMPROVED_SMOKE", "0") == "1"
    if not smoke_mode and (episode_count != 25 or start_episode != 0):
        raise ValueError(
            "improved evaluator is frozen to the official contiguous episode plan "
            "0..24; use EVAL_EPISODES=25 and START_EPISODE=0"
        )
    if smoke_mode and (episode_count < 1 or episode_count > 25 or start_episode != 0):
        raise ValueError("IMPROVED_SMOKE accepts 1..25 episodes from episode0")
    if os.environ.get("EVAL_SEED", "") or os.environ.get(
        "EPISODE_SELECTION_INPUT", ""
    ):
        raise ValueError(
            "improved evaluator forbids random/history episode selection; "
            "clear EVAL_SEED and EPISODE_SELECTION_INPUT"
        )
    if not os.path.isfile(step_config):
        raise FileNotFoundError(step_config)
    with open(step_config, "r", encoding="utf-8") as handle:
        step_limits = yaml.safe_load(handle)
    if not isinstance(step_limits, dict):
        raise ValueError("step-limit config must be a YAML mapping")
    missing_limits = [task for task in tasks if task not in step_limits]
    if missing_limits:
        raise ValueError("missing step limits: " + ", ".join(missing_limits))
    _check_fixed_dataset(dataset_root, tasks, episode_count)
    for path in (model_path, exp_cfg_path, mvt_cfg_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    paligemma_path = os.environ.get(
        "PALIGEMMA_PATH", os.path.join(project_root, "data/ckpt/paligemma")
    )
    if not os.path.isdir(paligemma_path):
        raise FileNotFoundError(paligemma_path)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "[config] tasks="
        + " ".join(tasks)
        + " episodes=0..24 output="
        + output_dir,
        flush=True,
    )
    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        eval_log_dir=output_dir,
        device=device,
    )
    run_evaluation(
        agent=agent,
        tasks=tasks,
        dataset_root=dataset_root,
        output_dir=output_dir,
        step_limits={task: int(step_limits[task]) for task in tasks},
        episode_count=episode_count,
        start_episode=start_episode,
        device=device,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
# ____
