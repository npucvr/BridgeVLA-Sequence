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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''

import csv
import pprint
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import RLBench.utils.peract_utils_rlbench as rlbench_utils
import GemBench.utils.peract_utils_gembench as gembench_utils
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con, aug_utils
from yarr.agents.agent import ActResult
from PIL import Image, ImageDraw
import torch
import numpy as np
import os


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    # Automatically detect color value range and convert to 0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    # Write to PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,  #  (3, 3, 224, 224) 
    gray_tensor: torch.Tensor,   #  (224, 224, 3) 
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug"
) -> None:
    """
    1. original_0.png, original_1.png, original_2.png   (original image)
    2. gray_0.png, gray_1.png, gray_2.png              (gray image)
    3. overlay_0.png, overlay_1.png, overlay_2.png     (transparent image + annotation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    color_imgs = color_tensor.cpu().numpy().transpose(0, 2, 3, 1) 
    gray_imgs = gray_tensor.cpu().numpy().transpose(2, 0, 1)     
    
    for i in range(3):

        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))
        

        gray_img = np.clip(gray_imgs[i], 0, 1) * 255
        gray_img = gray_img.astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))
        

        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img  
        rgba[..., 3] = 77            
        
    
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        
        max_pos = np.unravel_index(gray_imgs[i].argmax(), gray_imgs[i].shape)
        x = max_pos[1]  
        y = max_pos[0]  
        
      
        point_radius = 5
        draw.ellipse(
            [x-point_radius, y-point_radius, x+point_radius, y+point_radius],
            fill=(255, 0, 0, 255)  
        )
        
        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def apply_channel_wise_softmax(gray_tensor):
    """
    Apply softmax normalization independently to each grayscale channel
    Input shape: (H, W, C) -> Output shape: (H, W, C)
    All elements in each channel are processed by softmax and sum to 1
    """
    # Convert to PyTorch tensor (if not already)
    if not isinstance(gray_tensor, torch.Tensor):
        gray_tensor = torch.tensor(gray_tensor, dtype=torch.float32)
    
    # Separate each channel (C, H, W)
    channels = gray_tensor.permute(2, 0, 1)
    
    # Apply softmax to each channel and flatten
    softmax_channels = []
    for c in range(channels.shape[0]):
        channel = channels[c].flatten()
        softmax_channel = torch.softmax(channel, dim=0)
        softmax_channels.append(softmax_channel.view_as(channels[c]))
    
    # Merge channels and restore original shape (H, W, C)
    return torch.stack(softmax_channels, dim=2)


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    PMF_DIAGNOSTICS_STEP_FIELDS = [
        "mode", "task", "episode", "action_idx", "warmup", "success", "reward",
        "raw_x", "raw_y", "raw_z",
        "pmf_prior_x", "pmf_prior_y", "pmf_prior_z",
        "pmf_candidate_x", "pmf_candidate_y", "pmf_candidate_z",
        "executed_x", "executed_y", "executed_z",
        "raw_prior_x", "raw_prior_y", "raw_prior_z",
        "innovation_m", "candidate_correction_m", "executed_correction_m",
        "pmf_consistency_error_m", "history_prior_diff_m",
        "prev_raw_step_length_m", "raw_step_length_m", "step_length_ratio",
        "turn_angle_deg", "pred_gripper", "gripper_changed",
        "rot_qx", "rot_qy", "rot_qz", "rot_qw", "rotation_change_deg",
        "ee_before_x", "ee_before_y", "ee_before_z",
        "ee_after_x", "ee_after_y", "ee_after_z", "pred_collision",
    ]
    PMF_DIAGNOSTICS_EPISODE_FIELDS = [
        "mode", "task", "episode", "reward", "success", "episode_length",
        "num_actions", "language_goal",
    ]

    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS,
        cameras: list = rlbench_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        pmf_enabled: bool = False,
        pmf_prior_var: float = 9e-4,
        pmf_observation_var: float = 1e-4,
        pmf_diagnostics_enabled: bool = False,
        pmf_diagnostics_mode: str = None,
        pmf_diagnostics_log_dir: str = None,
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds
        self.cameras = cameras

        print("Cameras:",self.cameras)
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug
        if pmf_prior_var <= 0:
            raise ValueError("pmf_prior_var must be greater than 0")
        if pmf_observation_var <= 0:
            raise ValueError("pmf_observation_var must be greater than 0")
        self.pmf_enabled = pmf_enabled
        self.pmf_prior_var = pmf_prior_var
        self.pmf_observation_var = pmf_observation_var
        self.pmf_gain = pmf_prior_var / (pmf_prior_var + pmf_observation_var)
        self._pmf_prev_prev_wpt = None
        self._pmf_prev_wpt = None
        self.pmf_diagnostics_enabled = pmf_diagnostics_enabled
        self.pmf_diagnostics_mode = pmf_diagnostics_mode
        self.pmf_diagnostics_log_dir = pmf_diagnostics_log_dir
        self._diag_task = None
        self._diag_episode = None
        if self.pmf_diagnostics_enabled:
            if self.pmf_diagnostics_mode not in ("shadow", "pmf"):
                raise ValueError(
                    "pmf_diagnostics_mode must be 'shadow' or 'pmf' when diagnostics are enabled"
                )
            if self.pmf_diagnostics_mode == "shadow" and self.pmf_enabled:
                raise ValueError("shadow diagnostics require pmf_enabled=False")
            if self.pmf_diagnostics_mode == "pmf" and not self.pmf_enabled:
                raise ValueError("pmf diagnostics require pmf_enabled=True")
            if not self.pmf_diagnostics_log_dir:
                raise ValueError("pmf_diagnostics_log_dir is required when diagnostics are enabled")
            self._reset_pmf_diagnostics_episode_state()
        if self.pmf_enabled:
            print(
                f"[PMF] enabled | prior_var={self.pmf_prior_var:g} | "
                f"observation_var={self.pmf_observation_var:g} | "
                f"gain={self.pmf_gain:g}"
            )

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        params_to_optimize = filter(lambda p: p.requires_grad, self._network.parameters())

        self._optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self._lr,
            weight_decay=self._lambda_weight_l2,
        )


    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )


    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts



    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        tasks = replay_sample["tasks"]
        return_out = {}

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)  
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )


        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()    # Soft-label cross-entropy loss. The target has the same shape as the input and is no longer one-hot encoded, but represented by class probabilities.
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )


            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward() 
            self._optimizer.step()


            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)


        return return_out



    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 8)  
        

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3) 
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 

        action_grip = action_gripper_pose[:, -1].int()   # (b,)
        return_out = {}

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )
            import open3d as o3d
            def vis_pcd(pc, rgb,save_path):

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)  
                pcd.colors = o3d.utility.Vector3dVector(rgb) 
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
            
            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )

        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()  
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out


    @torch.no_grad()
    def act(
        self, step: int, observation: dict,visualize=False,visualize_save_dir="", return_gembench_action=False,
    ) -> ActResult:
        diag_ee_before = None
        if self.pmf_diagnostics_enabled:
            diag_ee_before = self._extract_diagnostics_ee_xyz(observation)
        language_goal =observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_ori = pc[0].clone()
        img_feat_ori=img_feat[0].clone()
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            language_goal=language_goal,
        )
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )            
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        diagnostics_step = None
        if self.pmf_diagnostics_enabled:
            diagnostics_step = self._prepare_pmf_diagnostics_step(
                raw_wpt=pred_wpt,
                pred_rot_quat=pred_rot_quat,
                pred_grip=pred_grip,
                pred_coll=pred_coll,
                ee_before=diag_ee_before,
            )
        pred_wpt = self._apply_prob_motion_filter(pred_wpt)
        if diagnostics_step is not None:
            self._finish_pmf_diagnostics_step(diagnostics_step, pred_wpt)
        if visualize:
            print("Visualizing")
            save_dir=visualize_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir=os.path.join(save_dir,f"step{str(step)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mvt1_img=out["mvt1_ori_img"][0,:,3:6]
            mvt2_img=out["mvt2_ori_img"][0,:,3:6]
            q_trans_1=q_trans[0,:,:3].clone().view(224,224,3)
            q_trans_2=q_trans[0,:,3:6].clone().view(224,224,3)
            q_trans_1=apply_channel_wise_softmax(q_trans_1)*100
            q_trans_2=apply_channel_wise_softmax(q_trans_2)*100
            visualize_images(mvt1_img,q_trans_1,save_dir=os.path.join(save_dir,"mvt1"))
            visualize_images(mvt2_img,q_trans_2,save_dir=os.path.join(save_dir,"mvt2"))
            save_point_cloud_with_color(os.path.join(save_dir,"point_cloud.ply"), pc_ori.cpu().numpy(), img_feat_ori.cpu().numpy(), pred_wpt[0].cpu().numpy())
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
                # [1.0],  # debug!!!!!!
            )
        )

        if return_gembench_action:
            continuous_action = np.concatenate(
                    [
                        pred_wpt[0].cpu().numpy(),
                        pred_rot_quat[0],
                        pred_grip[0].cpu().numpy(),
                    ], -1
                )
            return continuous_action
        else:
            return ActResult(continuous_action)



    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    def set_diagnostics_context(self, task_name: str, episode: int):
        if not self.pmf_diagnostics_enabled:
            return
        self._diag_task = task_name
        self._diag_episode = int(episode)

    def _reset_pmf_diagnostics_episode_state(self):
        self._diag_prev_prev_raw_wpt = None
        self._diag_prev_raw_wpt = None
        self._diag_prev_prev_shadow_wpt = None
        self._diag_prev_shadow_wpt = None
        self._diag_prev_rot_quat = None
        self._diag_prev_gripper = None
        self._diag_episode_rows = []

    @staticmethod
    def _extract_diagnostics_ee_xyz(observation):
        gripper_pose = observation.get("gripper_pose")
        if gripper_pose is None:
            return None
        if isinstance(gripper_pose, torch.Tensor):
            gripper_pose = gripper_pose.detach().cpu().numpy()
        pose = np.asarray(gripper_pose)
        if pose.size < 3:
            return None
        if pose.ndim == 1:
            xyz = pose[:3]
        else:
            if pose.shape[-1] < 3:
                return None
            xyz = pose.reshape(-1, pose.shape[-1])[-1, :3]
        xyz = np.asarray(xyz, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    @staticmethod
    def _diagnostics_rotation_change_deg(previous, current):
        if previous is None:
            return float("nan")
        previous = np.asarray(previous, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        previous_norm = np.linalg.norm(previous)
        current_norm = np.linalg.norm(current)
        if previous_norm <= 1e-12 or current_norm <= 1e-12:
            return float("nan")
        dot = abs(float(np.dot(previous / previous_norm, current / current_norm)))
        dot = float(np.clip(dot, 0.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(dot)))

    @staticmethod
    def _diagnostics_turn_angle_deg(v_prev, v_curr):
        prev_norm = float(np.linalg.norm(v_prev))
        curr_norm = float(np.linalg.norm(v_curr))
        if prev_norm <= 1e-12 or curr_norm <= 1e-12:
            return float("nan")
        cosine = float(np.dot(v_prev, v_curr) / (prev_norm * curr_norm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _diagnostics_xyz(prefix, value):
        if value is None:
            return {
                f"{prefix}_x": float("nan"),
                f"{prefix}_y": float("nan"),
                f"{prefix}_z": float("nan"),
            }
        value = np.asarray(value, dtype=np.float64)
        return {
            f"{prefix}_x": float(value[0]),
            f"{prefix}_y": float(value[1]),
            f"{prefix}_z": float(value[2]),
        }

    def _prepare_pmf_diagnostics_step(
        self, raw_wpt, pred_rot_quat, pred_grip, pred_coll, ee_before
    ):
        if self._diag_task is None or self._diag_episode is None:
            raise RuntimeError("diagnostics context must be set before agent.act()")

        if self._diag_episode_rows and ee_before is not None:
            self._diag_episode_rows[-1].update(
                self._diagnostics_xyz("ee_after", ee_before)
            )

        raw_tensor = raw_wpt.detach().clone()
        raw = raw_tensor[0].cpu().numpy().astype(np.float64)
        raw_prior_tensor = None
        if self._diag_prev_prev_raw_wpt is not None and self._diag_prev_raw_wpt is not None:
            raw_prior_tensor = (
                2 * self._diag_prev_raw_wpt - self._diag_prev_prev_raw_wpt
            )

        if self.pmf_diagnostics_mode == "shadow":
            history_prev_prev = self._diag_prev_prev_shadow_wpt
            history_prev = self._diag_prev_shadow_wpt
        else:
            history_prev_prev = self._pmf_prev_prev_wpt
            history_prev = self._pmf_prev_wpt

        warmup = history_prev_prev is None or history_prev is None
        prior_tensor = None
        candidate_tensor = raw_tensor.clone()
        if not warmup:
            prior_tensor = 2 * history_prev - history_prev_prev
            candidate_tensor = prior_tensor + self.pmf_gain * (
                raw_tensor - prior_tensor
            )

        if self.pmf_diagnostics_mode == "shadow":
            self._diag_prev_prev_shadow_wpt = self._diag_prev_shadow_wpt
            self._diag_prev_shadow_wpt = candidate_tensor.detach().clone()

        prior = None if prior_tensor is None else prior_tensor[0].cpu().numpy()
        candidate = candidate_tensor[0].cpu().numpy()
        raw_prior = (
            None
            if raw_prior_tensor is None
            else raw_prior_tensor[0].cpu().numpy()
        )

        raw_step_length = float("nan")
        prev_raw_step_length = float("nan")
        step_length_ratio = float("nan")
        turn_angle = float("nan")
        if self._diag_prev_raw_wpt is not None:
            v_curr = raw - self._diag_prev_raw_wpt[0].cpu().numpy()
            raw_step_length = float(np.linalg.norm(v_curr))
            if self._diag_prev_prev_raw_wpt is not None:
                v_prev = (
                    self._diag_prev_raw_wpt[0].cpu().numpy()
                    - self._diag_prev_prev_raw_wpt[0].cpu().numpy()
                )
                prev_raw_step_length = float(np.linalg.norm(v_prev))
                if prev_raw_step_length > 1e-12:
                    step_length_ratio = raw_step_length / (
                        prev_raw_step_length + 1e-12
                    )
                turn_angle = self._diagnostics_turn_angle_deg(v_prev, v_curr)

        rotation = np.asarray(pred_rot_quat[0], dtype=np.float64)
        gripper = int(pred_grip[0].detach().cpu().reshape(-1)[0].item())
        collision = int(pred_coll[0].detach().cpu().reshape(-1)[0].item())
        rotation_change = self._diagnostics_rotation_change_deg(
            self._diag_prev_rot_quat, rotation
        )
        gripper_changed = (
            0 if self._diag_prev_gripper is None else int(gripper != self._diag_prev_gripper)
        )

        innovation = float("nan") if prior is None else float(np.linalg.norm(raw - prior))
        candidate_correction = float(np.linalg.norm(candidate - raw))
        history_prior_diff = (
            float("nan")
            if prior is None or raw_prior is None
            else float(np.linalg.norm(prior - raw_prior))
        )

        row = {
            "mode": self.pmf_diagnostics_mode,
            "task": self._diag_task,
            "episode": self._diag_episode,
            "action_idx": len(self._diag_episode_rows),
            "warmup": int(warmup),
            "success": "",
            "reward": "",
            "innovation_m": innovation,
            "candidate_correction_m": candidate_correction,
            "executed_correction_m": float("nan"),
            "pmf_consistency_error_m": float("nan"),
            "history_prior_diff_m": history_prior_diff,
            "prev_raw_step_length_m": prev_raw_step_length,
            "raw_step_length_m": raw_step_length,
            "step_length_ratio": step_length_ratio,
            "turn_angle_deg": turn_angle,
            "pred_gripper": gripper,
            "gripper_changed": gripper_changed,
            "rot_qx": float(rotation[0]),
            "rot_qy": float(rotation[1]),
            "rot_qz": float(rotation[2]),
            "rot_qw": float(rotation[3]),
            "rotation_change_deg": rotation_change,
            "pred_collision": collision,
        }
        row.update(self._diagnostics_xyz("raw", raw))
        row.update(self._diagnostics_xyz("pmf_prior", prior))
        row.update(self._diagnostics_xyz("pmf_candidate", candidate))
        row.update(self._diagnostics_xyz("executed", None))
        row.update(self._diagnostics_xyz("raw_prior", raw_prior))
        row.update(self._diagnostics_xyz("ee_before", ee_before))
        row.update(self._diagnostics_xyz("ee_after", None))

        self._diag_prev_prev_raw_wpt = self._diag_prev_raw_wpt
        self._diag_prev_raw_wpt = raw_tensor.detach().clone()
        self._diag_prev_rot_quat = rotation.copy()
        self._diag_prev_gripper = gripper
        return {"row": row, "raw": raw, "candidate": candidate}

    def _finish_pmf_diagnostics_step(self, diagnostics_step, executed_wpt):
        row = diagnostics_step["row"]
        raw = diagnostics_step["raw"]
        candidate = diagnostics_step["candidate"]
        executed = executed_wpt[0].detach().cpu().numpy().astype(np.float64)
        row.update(self._diagnostics_xyz("executed", executed))
        row["executed_correction_m"] = float(np.linalg.norm(executed - raw))
        if self.pmf_diagnostics_mode == "pmf":
            row["pmf_consistency_error_m"] = float(
                np.linalg.norm(candidate - executed)
            )
        self._diag_episode_rows.append(row)

    @staticmethod
    def _append_diagnostics_csv(path, fieldnames, rows):
        has_header = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not has_header:
                writer.writeheader()
            writer.writerows(rows)

    def finalize_diagnostics_episode(
        self, reward, episode_length, language_goal, final_observation=None
    ):
        if not self.pmf_diagnostics_enabled:
            return
        if self._diag_task is None or self._diag_episode is None:
            raise RuntimeError("diagnostics context is missing at episode finalize")

        final_ee = (
            None
            if final_observation is None
            else self._extract_diagnostics_ee_xyz(final_observation)
        )
        if self._diag_episode_rows and final_ee is not None:
            self._diag_episode_rows[-1].update(
                self._diagnostics_xyz("ee_after", final_ee)
            )

        reward = float(reward)
        success = int(reward > 0)
        for row in self._diag_episode_rows:
            row["reward"] = reward
            row["success"] = success

        os.makedirs(self.pmf_diagnostics_log_dir, exist_ok=True)
        steps_path = os.path.join(
            self.pmf_diagnostics_log_dir, "pmf_diagnostics_steps.csv"
        )
        episodes_path = os.path.join(
            self.pmf_diagnostics_log_dir, "pmf_diagnostics_episodes.csv"
        )
        self._append_diagnostics_csv(
            steps_path,
            self.PMF_DIAGNOSTICS_STEP_FIELDS,
            self._diag_episode_rows,
        )
        episode_row = {
            "mode": self.pmf_diagnostics_mode,
            "task": self._diag_task,
            "episode": self._diag_episode,
            "reward": reward,
            "success": success,
            "episode_length": int(episode_length),
            "num_actions": len(self._diag_episode_rows),
            "language_goal": language_goal,
        }
        self._append_diagnostics_csv(
            episodes_path,
            self.PMF_DIAGNOSTICS_EPISODE_FIELDS,
            [episode_row],
        )
        self._diag_episode_rows = []

    def _apply_prob_motion_filter(self, pred_wpt: torch.Tensor) -> torch.Tensor:
        if not self.pmf_enabled:
            return pred_wpt

        filtered_wpt = pred_wpt.clone()
        if self._pmf_prev_prev_wpt is not None and self._pmf_prev_wpt is not None:
            prior_mean = 2 * self._pmf_prev_wpt - self._pmf_prev_prev_wpt
            filtered_wpt = prior_mean + self.pmf_gain * (
                filtered_wpt - prior_mean
            )

        self._pmf_prev_prev_wpt = self._pmf_prev_wpt
        self._pmf_prev_wpt = filtered_wpt.detach().clone()
        return filtered_wpt


    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans



    def reset(self):
        self._pmf_prev_prev_wpt = None
        self._pmf_prev_wpt = None
        if self.pmf_diagnostics_enabled:
            self._reset_pmf_diagnostics_episode_state()

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
