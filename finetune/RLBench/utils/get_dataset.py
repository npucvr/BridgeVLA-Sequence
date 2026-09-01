# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/get_dataset.py
import os
import shutil
import torch
import clip
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from dataset import create_replay, fill_replay
from peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from yarr.replay_buffer.sequence_replay_buffer import SequenceReplayBuffer
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def _resolve_episode_root(data_folder, split, task):
    """Resolve both official ``train/task`` and local task-root layouts."""
    candidates = [
        os.path.join(data_folder, split, task, "all_variations", "episodes"),
        os.path.join(data_folder, task, "all_variations", "episodes"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "RLBench episode directory not found; tried: " + ", ".join(candidates)
    )


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
    clip_cache_dir=None,
    sequence_training=False,
    sequence_length=1,
    burn_in_length=0,
    full_episode=False,
):

    sequence_training = bool(sequence_training)
    sequence_length = int(sequence_length)
    burn_in_length = int(burn_in_length)
    full_episode = bool(full_episode)
    if burn_in_length < 0:
        raise ValueError(
            f"burn_in_length must be non-negative, got {burn_in_length}"
        )
    if sequence_training and sequence_length < 2:
        raise ValueError(
            "sequence_length must be at least 2 when sequence_training is enabled"
        )
    if full_episode and burn_in_length:
        raise ValueError(
            "full_episode training starts at the episode boundary; "
            "burn_in_length must be zero"
        )

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )
    if not only_train:
        test_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )

    # load pre-trained language model
    try:
        clip_load_kwargs = {}
        if clip_cache_dir is not None:
            clip_load_kwargs["download_root"] = clip_cache_dir
        clip_model, _ = clip.load(
            "RN50", device="cpu", **clip_load_kwargs
        )  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    
    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        
        data_path_train = _resolve_episode_root(DATA_FOLDER, "train", task)
        data_path_val = (
            _resolve_episode_root(DATA_FOLDER, "val", task)
            if not only_train
            else None
        )
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
        )

        if not only_train:
            # print("----- Test Buffer -----")
            fill_replay(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # The legacy replay remains a one-step buffer.  Sequence training uses an
    # opt-in chronological sampler so existing replay files and the default
    # random transition path remain unchanged.
    if sequence_training:
        train_replay_source = SequenceReplayBuffer(
            train_replay_buffer,
            sequence_length=sequence_length,
            burn_in_length=burn_in_length,
            full_episode=full_episode,
        )
    else:
        train_replay_source = train_replay_buffer

    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_source,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset


