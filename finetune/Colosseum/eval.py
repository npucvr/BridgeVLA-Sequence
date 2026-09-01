'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/eval.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import yaml
import csv
import torch
import cv2
import shutil
# gbw____
import numbers
# ____
import numpy as np
from omegaconf import OmegaConf
from multiprocessing import Value
# gbw____
# summary_iterator is not used by this evaluator.  Keep the optional import
# for environments that provide TensorFlow, but do not make Colosseum A0
# depend on an unrelated TensorFlow installation.
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
except ModuleNotFoundError:
    summary_iterator = None
# ____
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.log_writer import LogWriter
from yarr.agents.agent import VideoSummary

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from utils.peract_utils_colosseum import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    get_official_peract,
)
from utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from bridgevla.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    COLOSSEUM_TASKS,
)
from bridgevla.utils.rvt_utils import load_agent as load_agent_state

from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_PY_FOLDER,
    TASKS_TTM_FOLDER,
)

from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)


def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False):
    device = f"cuda:{device}"


    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    # for rvt-2 we do not change place_with_mean regardless of the arg
    # done this way to ensure backward compatibility and allow the
    # flexibility for rvt-1
    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    rvt = MVT(
        renderer_device=device,
        **mvt_cfg,
    )

    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()


    print("Agent Information")
    print(agent)
    return agent


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
):
    agent.eval()
    if isinstance(agent, bridgevla_agent.RVTAgent):
        # agent.load_clip()
        pass
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = COLOSSEUM_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    
    task_class_variation_idx = []
    task_class_base = []
    for task in tasks:
        task_class_base.append('_'.join(task.split('_')[:-1]))
        if task_class_base[-1] not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = name_to_class(task_class_base[-1], TASKS_PY_FOLDER) # task_file_to_task_class(task_class_base)
        task_class_variation_idx.append(int(task.split('_')[-1]))
        task_classes.append(task_class)

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
        base_cfg_name=task_class_base,
        task_class_variation_idx=task_class_variation_idx,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        language_goals=[]
        total_count=0
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=replay_ground_truth,
            )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
                total_count+=1
            except StopIteration as e:
                assert False
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Total Episode {total_count} |Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        # gbw____
        # YARR 的 _SimpleAccumulator.pop() 只有在累计超过一个完整 episode
        # 时才返回统计结果。官方 workaround 为了逐 trial 重试而使用
        # eval_episodes=1，此时 episode 已经完成，但 pop() 会返回空列表，
        # 进而把真实 reward 错误地写成 unknown。peak() 只读取当前已完成
        # episode 的统计；随后 reset() 保证本进程不会把本 task 的状态带到
        # 后续 task。该兼容逻辑不改变 rollout、动作或 reward。
        if not summaries and task_rewards:
            summaries.extend(stats_accumulator.peak())
            stats_accumulator.reset()
        # ____
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            # gbw____
            # 多个 repeat 并行时不能共享 cwd 下的 ./tmp；否则不同进程会
            # 同时删除/覆盖同一个 task 的帧和 palette。每个 eval 的 log_dir
            # 已按 repeat/task 隔离，因此把视频临时文件放到该目录下。
            video_tmp_root = os.path.join(log_dir or ".", "video_tmp")
            video_image_folder = os.path.join(video_tmp_root, task_name)
            palette_image_folder = os.path.join(video_tmp_root, "palette_folder")
            # ____
            palette_image_path=os.path.join(palette_image_folder,"palette.png")
            num_succ_video = 25
            num_fail_video = 25
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = language_goals.pop(0)
                    lang_goal=lang_goal.replace(" ", "_")
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if (task_rewards[video_cnt] > 99 and video_success_cnt < num_succ_video) or \
                        (not task_rewards[video_cnt] > 99 and video_fail_cnt < num_fail_video):
                        if task_rewards[video_cnt] > 99:
                            video_path = os.path.join(
                                record_folder,
                                f"{task_name}_{lang_goal}_success_{video_success_cnt}.mp4",
                            )
                            video_success_cnt += 1
                        else:
                            video_path = os.path.join(
                                record_folder, f"{task_name}_{lang_goal}_fail_{video_fail_cnt}.mp4"
                            )
                            video_fail_cnt += 1
                        video_cnt += 1
                        os.makedirs(video_image_folder, exist_ok=True)
                        os.makedirs(palette_image_folder, exist_ok=True)
                        for idx in range(len(video) - 10):
                            cv2.imwrite(
                                os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                            )
                        images_path = os.path.join(video_image_folder, r"%d.png")
                        os.system(
                            "ffmpeg -i {} -vf palettegen {} -hide_banner -loglevel error".format(
                                images_path, palette_image_path
                            )
                        )
                        
                        os.system(
                            "ffmpeg -framerate {} -i {} -i {} -lavfi paletteuse {} -hide_banner -loglevel error".format(
                                record_fps, images_path, palette_image_path, video_path
                            )
                        )

                        print(f'video saved - {task_name}')
                        # gbw____
                        # ffmpeg 失败时 palette 可能不存在；清理只能作用于
                        # 当前进程自己的临时目录，不能因清理异常中断整个 eval。
                        # ____
                        if os.path.exists(palette_image_path):
                            os.remove(palette_image_path)
                        shutil.rmtree(video_image_folder, ignore_errors=True)

    eval_env.shutdown()

    if logging:
        csv_fp.close()

    # set agent to back train mode
    agent.train()


    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):
    assert args.model_name is not None
    model_path=os.path.join(args.model_folder, args.model_name)
    tb = TensorboardManager(args.eval_log_dir)
    tasks_to_eval = deepcopy(args.tasks)

    model_idx = get_model_index(model_path)
    if model_idx is None:
        model_idx = 0

    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=args.eval_log_dir,
        device=args.device,
        use_input_place_with_mean=args.use_input_place_with_mean
    )

    agent_eval_log_dir = os.path.join(
        args.eval_log_dir, os.path.basename(model_path).split(".")[0]
    )

    os.makedirs(agent_eval_log_dir, exist_ok=True)
    scores = eval(
        agent=agent,
        tasks=tasks_to_eval,
        eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        replay_ground_truth=args.ground_truth,
        device=args.device,
        headless=args.headless,
        logging=True,
        log_dir=agent_eval_log_dir,
        verbose=True,
        save_video=args.save_video,
    )
    print(f"model {model_path}, scores {scores}")
    task_scores = {}
    for i in range(len(tasks_to_eval)):
        task_scores[tasks_to_eval[i]] = scores[i]
    print("avg score: ", task_scores)
    # gbw____
    # 某个 episode 若确实在 reset 阶段失败，task_score 可能为
    # ``unknown``。它应由 workaround 作为 invalid trial 重试，而不是
    # 让 TensorBoard 的 float() 转换再次把整个评测进程打崩。
    numeric_task_scores = {
        key: value
        for key, value in task_scores.items()
        if isinstance(value, numbers.Real)
        and np.isfinite(float(value))
    }
    dropped_task_scores = [
        key for key in task_scores if key not in numeric_task_scores
    ]
    if dropped_task_scores:
        print(
            "[Evaluation] no numeric score for "
            f"{dropped_task_scores}; these trials are not valid workaround samples."
        )
    tb.update("eval", model_idx, numeric_task_scores)
    # ____
    tb.writer.flush()

    tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(args.model_folder, args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)
