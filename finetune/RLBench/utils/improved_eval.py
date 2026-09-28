"""Evaluator-side repairs aligned with BridgeVLA++ (dev-bowen eval_improved).

These changes do not modify the policy or checkpoint.  They only make the
held-out evaluation protocol more faithful:

* ghost renderable asset repair for known variation-specific objects;
* correct routing of the 9-D action collision flag into the arm action mode;
* velocity-based arm settling around gripper execution;
* workspace clipping of translation targets.

All repairs are gated by ``IMPROVED_*`` environment variables and default on.
"""

from __future__ import annotations

import os

import numpy as np
from pyrep.objects.shape import Shape
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.backend.scene import Scene

# Renderable-only flags.  No physics, success criterion, or policy input change.
GHOST_RENDERABLE_FIX = {
    "put_item_in_drawer": ["item"],
    "put_money_in_safe": ["dollar_front_visual", "dollar_back_visual"],
    "light_bulb_in": ["lamp_base", "lamp_screw"],
    "place_cups": ["place_cups_holder_spoke3"],
    "place_shape_in_shape_sorter": ["shape_sorter_top"],
}


def improved_asset_fix_enabled() -> bool:
    return os.environ.get("IMPROVED_ASSET_FIX", "1") == "1"


def improved_settle_enabled() -> bool:
    return os.environ.get("IMPROVED_SETTLE", "1") == "1"


def force_renderable_ghosts(task) -> int:
    """Force known ghost objects renderable; re-capture uses caller."""
    if not improved_asset_fix_enabled():
        return 0
    try:
        task_name = task._task.get_name()
    except Exception:
        try:
            task_name = task.get_name()
        except Exception:
            return 0
    count = 0
    for object_name in GHOST_RENDERABLE_FIX.get(task_name, []):
        try:
            Shape(object_name).set_renderable(True)
            count += 1
        except Exception:
            continue
    return count


class ImprovedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
    """Clip translation targets into the RLBench workspace before planning."""

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.asarray(
                [
                    scene._workspace_minx,
                    scene._workspace_miny,
                    scene._workspace_minz,
                ]
            )
            + 1e-7,
            np.asarray(
                [
                    scene._workspace_maxx,
                    scene._workspace_maxy,
                    scene._workspace_maxz,
                ]
            )
            - 1e-7,
        )
        return super().action(scene, action, ignore_collisions)


class ImprovedMoveArmThenGripper(MoveArmThenGripper):
    """Route [pose(7), gripper(1), collision(1)] and settle around gripper."""

    def _wait_arm_stopped(self, scene: Scene) -> None:
        if not improved_settle_enabled():
            return
        max_velocity = float(os.environ.get("IMPROVED_SETTLE_MAX_VEL", "0.01"))
        max_steps = int(os.environ.get("IMPROVED_SETTLE_MAX_STEPS", "100"))
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
        gripper_action = np.asarray(action[arm_size : arm_size + gripper_size])
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
