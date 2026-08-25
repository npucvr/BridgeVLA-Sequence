#!/usr/bin/env python3
"""Cache frozen BridgeVLA Stage-1 tokens at RLBench keyframes."""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import torch

from bridgevla.mvt.config import get_cfg_defaults
from bridgevla.mvt.mvt import MVT
from bridgevla.mvt import utils as mvt_utils
from bridgevla.utils import rvt_utils
from bridgevla.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from bridgevla.libs.peract.helpers.utils import extract_obs
from RLBench.utils import peract_utils_rlbench as rlbench_cfg
from peract_colab.rlbench.utils import get_stored_demo


TASKS = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paligemma_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="number of episodes; default caches every episode found",
    )
    parser.add_argument("--encode_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_network(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file("finetune/bridgevla/mvt/configs/rvt2.yaml")
    cfg.paligemma_path = args.paligemma_path
    cfg.stage1_history_len = 1

    network = MVT(
        renderer_device=args.device,
        load_pretrain=False,
        pretrain_path=None,
        **cfg,
    ).to(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    state = (
        checkpoint.get("model_state", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    missing, unexpected = network.load_state_dict(state, strict=False)
    missing = [
        key
        for key in missing
        if not key.startswith("mvt1.stage1_token_adapter.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad = False
    return network


def make_replay_sample(obs, device):
    obs_dict = extract_obs(
        copy.copy(obs),
        rlbench_cfg.CAMERAS,
        t=0,
        episode_length=25,
    )
    sample = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            sample[key] = torch.from_numpy(value).unsqueeze(0).unsqueeze(0).to(device)
    return sample


def encode_keypoint_batch(network, demo, frames, description, device):
    samples = [make_replay_sample(demo[frame], device) for frame in frames]
    # All samples have the same keys and shapes; stack the singleton batch/time
    # dimensions to obtain the replay layout [B, T=1, ...].
    replay_sample = {
        key: torch.cat([sample[key] for sample in samples], dim=0)
        for key in samples[0]
    }
    obs, pcd = rlbench_cfg._preprocess_inputs(replay_sample, rlbench_cfg.CAMERAS)
    with torch.inference_mode():
        pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc,
            img_feat,
            rlbench_cfg.SCENE_BOUNDS,
            no_op=False,
        )
        pc = [
            mvt_utils.place_pc_in_cube(
                point_cloud,
                with_mean_or_bounds=False,
                scene_bounds=rlbench_cfg.SCENE_BOUNDS,
            )[0]
            for point_cloud in pc
        ]
        rendered = network.render(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,
            mvt1_or_mvt2=True,
            dyn_cam_info=None,
        )
        language_goal = [[[description]] for _ in frames]
        output = network.mvt1(
            img=rendered,
            language_goal=language_goal,
            forward_no_feat=True,
            return_stage1_tokens=True,
        )
    return output["stage1_tokens"].to(dtype=torch.bfloat16).cpu()


def episode_root(data_root, task):
    candidates = [
        Path(data_root) / task / "all_variations" / "episodes",
        Path(data_root) / "train" / task / "all_variations" / "episodes",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "RLBench episode directory not found; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def cache_episode(network, data_root, output_root, task, episode_idx, args):
    episode_dir = episode_root(data_root, task) / f"episode{episode_idx}"
    output_path = Path(output_root) / task / f"episode{episode_idx}.pt"
    if output_path.exists() and not args.overwrite:
        print(f"skip existing {output_path}")
        return

    demo = get_stored_demo(str(episode_dir.parent), episode_idx)
    keypoint_frames = keypoint_discovery(demo)
    with (episode_dir / "variation_descriptions.pkl").open("rb") as file:
        description = pickle.load(file)[0]

    chunks = []
    for start in range(0, len(keypoint_frames), args.encode_batch_size):
        frames = keypoint_frames[start : start + args.encode_batch_size]
        print(
            f"{task}/episode{episode_idx}: "
            f"encoding frames {start}:{start + len(frames)} / {len(keypoint_frames)}",
            flush=True,
        )
        chunks.append(
            encode_keypoint_batch(
                network,
                demo,
                frames,
                description,
                args.device,
            )
        )

    tokens = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    torch.save(
        {
            "keypoint_frames": torch.tensor(keypoint_frames, dtype=torch.long),
            "tokens": tokens,
        },
        temporary_path,
    )
    temporary_path.replace(output_path)
    print(f"wrote {output_path} tokens={tuple(tokens.shape)}", flush=True)


def episode_indices(args, task):
    if args.num_episodes is not None:
        return range(args.episode_start, args.episode_start + args.num_episodes)

    episode_root_path = episode_root(args.data_root, task)
    indices = []
    for path in episode_root_path.iterdir():
        if not path.is_dir() or not path.name.startswith("episode"):
            continue
        suffix = path.name[len("episode") :]
        if suffix.isdigit():
            indices.append(int(suffix))
    indices = sorted(index for index in indices if index >= args.episode_start)
    if not indices:
        raise FileNotFoundError(f"no episodes found under {episode_root_path}")
    return indices


def main():
    args = parse_args()
    network = load_network(args)
    for task in args.tasks:
        for episode_idx in episode_indices(args, task):
            cache_episode(
                network,
                args.data_root,
                args.output_root,
                task,
                episode_idx,
                args,
            )


if __name__ == "__main__":
    main()
