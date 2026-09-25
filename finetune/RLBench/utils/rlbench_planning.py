#Copy from https://github.com/NVlabs/RVT/blob/master/rvt/utils/rlbench_planning.py
import numpy as np
from pyrep.errors import ConfigurationPathError
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)
from rlbench.backend.exceptions import InvalidActionError


MAX_PLANNING_RETRIES = 2


def _configuration_path_cause(error):
    """Find a wrapped planner error without retrying unrelated invalid actions."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ConfigurationPathError):
            return current
        pending.extend((current.__cause__, current.__context__))
    return None


def _scene_task_name(scene):
    task = getattr(scene, "task", None)
    return type(task).__name__ if task is not None else type(scene).__name__


class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, planning_retries=0, **kwargs):
        if (
            not isinstance(planning_retries, int)
            or not 0 <= planning_retries <= MAX_PLANNING_RETRIES
        ):
            raise ValueError(
                f"planning_retries must be an integer in [0, {MAX_PLANNING_RETRIES}]"
            )
        self._planning_retries = planning_retries
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        # Preserve the caller's prediction and apply the existing workspace clamp
        # once; every bounded planner attempt receives an identical copy.
        action = np.array(action, copy=True)
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        max_attempts = self._planning_retries + 1
        failed_path_attempts = 0
        for attempt in range(1, max_attempts + 1):
            try:
                super().action(scene, action.copy(), ignore_collisions)
                if failed_path_attempts:
                    print(
                        "[PLANNING_RETRY] "
                        f"task={_scene_task_name(scene)} "
                        f"target={np.array2string(action[:7], precision=5)} "
                        f"result=recovered after={failed_path_attempts}"
                    )
                return
            except InvalidActionError as error:
                path_error = _configuration_path_cause(error)
                if path_error is None:
                    if self._planning_retries:
                        print(
                            "[PLANNING_RETRY] "
                            f"task={_scene_task_name(scene)} "
                            f"target={np.array2string(action[:7], precision=5)} "
                            f"result=non_retryable_invalid_action "
                            f"error={type(error).__name__}: {error}"
                        )
                    raise

                failed_path_attempts += 1
                print(
                    "[PLANNING_RETRY] "
                    f"task={_scene_task_name(scene)} "
                    f"target={np.array2string(action[:7], precision=5)} "
                    f"attempt={attempt}/{max_attempts} result=ConfigurationPathError "
                    f"cause={path_error}"
                )
                if attempt == max_attempts:
                    print(
                        "[PLANNING_RETRY] "
                        f"task={_scene_task_name(scene)} "
                        f"result=exhausted retries={self._planning_retries}; re-raising"
                    )
                    raise
