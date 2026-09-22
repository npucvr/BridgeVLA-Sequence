import logging
from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
# gbw____
# Colosseum 归档采用“base_task/task_variant/variation0/episodes”的布局，
# 并且不保存 overhead 相机目录；项目内的 Colosseum loader 已针对这个
# 布局做了最小适配。不能调用通用 RLBench loader。
from utils import colosseum_utils as utils
# ____
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError, \
    TaskEnvironmentError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from rlbench.task_environment import TaskEnvironment

_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 10


class TaskEnvironmentExt(TaskEnvironment):

    def __init__(self, *args, **kwargs):
        super(TaskEnvironmentExt, self).__init__(*args, **kwargs)

    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                               or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            # gbw____
            # Colosseum evaluation archives in this workspace contain the
            # low-dimensional demo state and four rendered-camera folders,
            # but no RLBench ``overhead_*`` folders.  Evaluation only needs
            # the demo state to restore the scene; current observations are
            # rendered by CoppeliaSim from the active ObservationConfig.
            # Keeping image loading disabled also avoids changing the shared
            # RLBench loader used by the other benchmarks.
            # ____
            # gbw____
            # image_paths 已由当前调用方控制；Colosseum loader 与本项目
            # RLBench fork 的签名一致，不接受不存在的 load_images 关键字。
            # ____
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.task_path, self._obs_config,
                random_selection, from_episode_number)
        else:
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            demos = self._get_live_demos(
                amount, callable_each_step, max_attempts)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
        return demos

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                try:
                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e) + ' Attempts left: ' + str(attempts))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos

    def reset_to_demo(self, demo: Demo) -> (List[str], Observation):
        demo.restore_state()
        return self.reset()
