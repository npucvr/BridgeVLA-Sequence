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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import random
import subprocess
import time
import tqdm
import yaml
import argparse
import time
from collections import defaultdict
from contextlib import redirect_stdout
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from bridgevla.mvt.lora import (
    inject_discrete_action_lora,
    merged_lora_state_dict,
)
from utils.get_dataset import get_dataset
from bridgevla.utils.rvt_utils import (
    get_num_feat,
    RLBENCH_TASKS,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    TRAIN_REPLAY_STORAGE_DIR,
)

def train(
    agent,
    dataset,
    training_iterations,
    epoch,
    rank=0,
    sequence_training=False,
    sequence_bptt_length=1,
):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
        dist.barrier()
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "replay_sample": batch,
            "backprop": True,
            "reset_log": (iteration == 0),
        }
        if sequence_training:
            update_args["bptt_length"] = sequence_bptt_length
        has_sequence_batch = "valid_mask" in batch
        if has_sequence_batch != sequence_training:
            raise RuntimeError(
                "dataset/trainer sequence mode mismatch: "
                f"configured={sequence_training}, batch_has_valid_mask={has_sequence_batch}"
            )
        if sequence_training:
            out = agent.update_sequence(**update_args)
        else:
            out = agent.update(**update_args)
        dist.barrier()
        if rank == 0:
            step=epoch*training_iterations+iteration
            wandb.log(
                    out,
                    step=step,
                )
    return log

def save_agent(agent, path, epoch):
    model = agent._network

    if isinstance(model, DDP):
        model = model.module
    else:
        model = model
    # LoRA is a training-time parameterization. Export a standard BridgeVLA
    # checkpoint by merging its deltas into the original action-head weights.
    model_state = merged_lora_state_dict(model)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
        },
        path,
    )



def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks



def get_time():
    import datetime
    now = datetime.datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    #  'MM-DD-HH-MM'
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name


def get_logdir(cmd_args, exp_cfg,dist):
    log_dir = os.path.join(cmd_args.log_dir,"train" ,exp_cfg.exp_id,cmd_args.exp_note)
    if cmd_args.debug==True:
        log_dir = os.path.join(log_dir,"debug")

    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    trial_time=get_time()
    log_dir = os.path.join(log_dir,f"{trial_time}")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)



def load_initial_checkpoint(backbone, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model_state", checkpoint)
    source_state = dict(state)
    legacy_hidden_route_prefixes = (
        "mvt1.A_psi.",
        "mvt1.U_omega.",
        "mvt1.observation_decoder.",
    )
    hidden_route_prefixes = (
        "mvt1.F_phi.",
        "mvt1.filter_correction.",
    ) + legacy_hidden_route_prefixes
    legacy_direct_adapter_prefixes = (
        "mvt1.hidden_state_to_feat.",
        "mvt1.hidden_state_to_trans.",
    )
    # Older hidden-state checkpoints contained direct action-head adapters and
    # the removed prior-observation route. Ignore those legacy keys when
    # initializing the current model.
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith(legacy_direct_adapter_prefixes)
    }
    if not backbone.mvt1.hidden_state_enabled:
        # A route checkpoint can still initialize the released policy when the
        # optional route is disabled; its extra keys are intentionally ignored.
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(hidden_route_prefixes)
        }
    else:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(legacy_hidden_route_prefixes)
        }
    continuous_rotation = bool(
        getattr(backbone.mvt1, "continuous_rotation", False)
    )
    rot_ver = int(getattr(backbone.mvt1, "rot_ver", 0))
    rot_6d = rot_ver == 2
    if continuous_rotation:
        # The new 6D head intentionally replaces the three 72-way Euler
        # heads.  Keep the shared backbone/action features from an old filter
        # checkpoint while allowing the replacement head to initialize fresh.
        legacy_rotation_prefixes = (
            "mvt1.feat_fc_pe.",
            "mvt1.feat_fc_x.",
            "mvt1.feat_fc_y.",
            "mvt1.feat_fc_z.",
        )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(legacy_rotation_prefixes)
        }
    if rot_6d:
        # BridgeVLA++ rot_ver=2 replaces the discrete rot/grip/coll heads with
        # one feat_fc (feat_dim=10). Drop the legacy heads (including BN) so
        # load_state_dict does not see them as unexpected.
        legacy_rotation_prefixes = (
            "mvt1.feat_fc_pe.",
            "mvt1.feat_fc_x.",
            "mvt1.feat_fc_y.",
            "mvt1.feat_fc_z.",
            "mvt1.feat_fc_ex_rot.",
            "mvt1.feat_fc_init_bn.",
        )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(legacy_rotation_prefixes)
        }
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    unexpected = list(unexpected)
    allowed_missing_prefixes = []
    if backbone.mvt1.hidden_state_enabled:
        allowed_missing_prefixes.extend(
            ("mvt1.F_phi.", "mvt1.filter_correction.")
        )
    if continuous_rotation:
        allowed_missing_prefixes.append("mvt1.feat_fc_rot6d.")
    if rot_6d:
        allowed_missing_prefixes.append("mvt1.feat_fc.")
    allowed_missing_prefixes = tuple(allowed_missing_prefixes)
    missing = [
        key
        for key in missing
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            "Initial checkpoint is incompatible with the BridgeVLA model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if continuous_rotation:
        _initialize_continuous_rotation_head(backbone, source_state)
    if rot_6d:
        _initialize_rot6d_feat_fc(backbone, source_state)
    print(
        f"Loaded initial checkpoint: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')})"
    )


def _initialize_continuous_rotation_head(backbone, source_state):
    """Warm-start the replacement 6D head near identity rotation.

    Hidden layers are averaged from the old x/y/z heads. The final projection is
    zero-initialised with a bias of the identity ortho6d (columns e1,e2), so the
    head starts at a valid rotation instead of random pose noise.
    """
    head = backbone.mvt1.feat_fc_rot6d
    axes = ("x", "y", "z")
    with torch.no_grad():
        for layer_index in (0, 2):
            weight_keys = [
                f"mvt1.feat_fc_{axis}.{layer_index}.weight" for axis in axes
            ]
            bias_keys = [
                f"mvt1.feat_fc_{axis}.{layer_index}.bias" for axis in axes
            ]
            if all(key in source_state for key in weight_keys):
                head[layer_index].weight.copy_(
                    torch.stack([source_state[key] for key in weight_keys]).mean(0)
                )
            if all(key in source_state for key in bias_keys):
                head[layer_index].bias.copy_(
                    torch.stack([source_state[key] for key in bias_keys]).mean(0)
                )
        # Final linear is index 4 in get_feat_fc. Start at identity rotation.
        final = head[4]
        if isinstance(final, torch.nn.Linear) and final.out_features == 6:
            final.weight.zero_()
            final.bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=final.bias.dtype)
            )
    print(
        "Initialized continuous 6D head: averaged hidden layers from discrete "
        "rotation heads, final projection set to identity ortho6d."
    )


def _initialize_rot6d_feat_fc(head_owner, source_state):
    """Warm-start BridgeVLA++ ``feat_fc`` (rot_ver=2, feat_dim=10) from model_80.

    Layout: [6D rot | grip(2) | collision(2)]. Hidden layers are averaged from
    the discrete x/y/z heads and feat_fc_ex_rot. The rotation slice of the final
    projection is zero-initialised with bias = identity ortho6d (columns e1,e2);
    the grip/collision slice is copied from feat_fc_ex_rot so those logits stay
    calibrated.
    """
    head = head_owner.mvt1.feat_fc
    axes = ("x", "y", "z")
    with torch.no_grad():
        for layer_index in (0, 2):
            weight_keys = [
                f"mvt1.feat_fc_{axis}.{layer_index}.weight" for axis in axes
            ] + ["mvt1.feat_fc_ex_rot.%d.weight" % layer_index]
            bias_keys = [
                f"mvt1.feat_fc_{axis}.{layer_index}.bias" for axis in axes
            ] + ["mvt1.feat_fc_ex_rot.%d.bias" % layer_index]
            weight_keys = [k for k in weight_keys if k in source_state]
            bias_keys = [k for k in bias_keys if k in source_state]
            if weight_keys:
                head[layer_index].weight.copy_(
                    torch.stack([source_state[k] for k in weight_keys]).mean(0)
                )
            if bias_keys:
                head[layer_index].bias.copy_(
                    torch.stack([source_state[k] for k in bias_keys]).mean(0)
                )
        # Final linear is index 4 in get_feat_fc.
        final = head[4]
        if isinstance(final, torch.nn.Linear) and final.out_features == 10:
            final.weight.zero_()
            final.bias.zero_()
            final.bias[:6].copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=final.bias.dtype)
            )
            ex_rot_key = "mvt1.feat_fc_ex_rot.4.weight"
            ex_rot_bias_key = "mvt1.feat_fc_ex_rot.4.bias"
            if ex_rot_key in source_state:
                final.weight[6:].copy_(source_state[ex_rot_key])
            if ex_rot_bias_key in source_state:
                final.bias[6:].copy_(source_state[ex_rot_bias_key])
    print(
        "Initialized BridgeVLA++ rot_ver=2 feat_fc: hidden averaged from "
        "discrete heads, rotation final = identity ortho6d, grip/coll copied "
        "from feat_fc_ex_rot."
    )


def freeze_for_hidden_state_route(
    backbone,
    unfreeze_action_path=False,
    discrete_action_lora=False,
    lora_rank=8,
    lora_alpha=16,
):
    """Freeze BridgeVLA and train the filter or a selected action path."""
    if not backbone.mvt1.hidden_state_enabled:
        raise ValueError(
            "hidden_state_route_only requires hidden_state_enabled=True"
        )
    if unfreeze_action_path and discrete_action_lora:
        raise ValueError(
            "Choose either full action-path unfreezing or discrete-head LoRA"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    if discrete_action_lora:
        if int(getattr(backbone.mvt1, "rot_ver", 0)) != 1:
            raise ValueError(
                "--hidden_state_lora_discrete_action_path requires mvt.rot_ver=1"
            )
        targets = inject_discrete_action_lora(
            backbone.mvt1, rank=lora_rank, alpha=lora_alpha
        )
        trainable_count = sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        )
        if trainable_count == 0:
            raise RuntimeError("discrete action LoRA injected no trainable parameters")
        print(
            "Training discrete action LoRA only (filter and base model frozen): "
            f"targets={targets} rank={lora_rank} alpha={lora_alpha} "
            f"trainable={trainable_count}"
        )
        return

    trainable_modules = [
        backbone.mvt1.F_phi,
        backbone.mvt1.filter_correction,
    ]
    if unfreeze_action_path:
        continuous_rotation = bool(
            getattr(backbone.mvt1, "continuous_rotation", False)
        )
        rot_ver = int(getattr(backbone.mvt1, "rot_ver", 0))
        if continuous_rotation:
            trainable_modules.extend(
                [
                    backbone.mvt1.feat_fc_init_bn,
                    backbone.mvt1.feat_fc_rot6d,
                    backbone.mvt1.up0.net_out[4],
                    backbone.mvt1.up0.net_mask[2],
                ]
            )
        elif rot_ver == 2:
            # BridgeVLA++ 6D head + the convex-upsample output tails.
            trainable_modules.extend(
                [
                    backbone.mvt1.feat_fc,
                    backbone.mvt1.up0.net_out[4],
                    backbone.mvt1.up0.net_mask[2],
                ]
            )
        else:
            raise ValueError(
                "--hidden_state_unfreeze_action_path requires "
                "mvt.continuous_rotation=True or mvt.rot_ver=2"
            )
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable_names = [
        module.__class__.__name__ for module in trainable_modules
    ]
    if unfreeze_action_path:
        print(
            "Training filter plus finite action path: "
            + ", ".join(trainable_names)
        )
    else:
        print("Training only hidden-state modules: " + ", ".join(trainable_names))


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print("Can not find RANK and WORLD_SIZE, Debug Mode")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "9001")
            os.environ.setdefault("LOCAL_RANK", "0")
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def seed_training(seed, rank):
    """Seed Python, NumPy, and Torch before dataset/model construction.

    The filter projection has its own frozen seed, but the trainable route and
    replay sampling also need an explicit seed for matched ablations.  Offset
    the process-local streams by rank while keeping single-GPU runs exactly
    reproducible from the requested base seed.
    """
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Training seed: base={int(seed) - int(rank)} rank={int(rank)} effective={seed}")



def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)
    if cmd_args.seed is not None:
        seed_training(cmd_args.seed, dist.get_rank())
    exp_cfg = exp_cfg_mod.get_cfg_defaults()

    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    ddp = int(os.environ['WORLD_SIZE']) > 1
    print(f"Total devices: {dist.get_world_size()}")
    if ddp:
        print(f"Running DDP on rank {dist.get_rank()}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    NUM_TRAIN = cmd_args.num_train
    sequence_training = bool(
        exp_cfg.hidden_state_sequence_training
        or cmd_args.hidden_state_sequence_training
    )
    sequence_length = int(
        cmd_args.hidden_state_sequence_length
        if cmd_args.hidden_state_sequence_length is not None
        else exp_cfg.hidden_state_sequence_length
    )
    bptt_length = int(
        cmd_args.hidden_state_sequence_bptt_length
        if cmd_args.hidden_state_sequence_bptt_length is not None
        else exp_cfg.hidden_state_sequence_bptt_length
    )
    full_episode = bool(
        exp_cfg.hidden_state_sequence_full_episode
        or cmd_args.hidden_state_sequence_full_episode
    )
    burn_in_length = int(
        cmd_args.hidden_state_sequence_burn_in
        if cmd_args.hidden_state_sequence_burn_in is not None
        else exp_cfg.hidden_state_sequence_burn_in
    )
    if bptt_length < 1:
        raise ValueError(
            "hidden_state_sequence_bptt_length must be at least 1"
        )
    if burn_in_length < 0:
        raise ValueError(
            "hidden_state_sequence_burn_in must be non-negative"
        )
    if full_episode and burn_in_length:
        raise ValueError(
            "full-episode sequence training starts at the episode boundary; "
            "hidden_state_sequence_burn_in must be zero"
        )
    if sequence_training and sequence_length < 2:
        raise ValueError(
            "hidden_state_sequence_length must be at least 2 when "
            "sequence training is enabled"
        )
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * dist.get_world_size()))

    if exp_cfg.epochs!=cmd_args.epochs:
        print(f"cmd args epochs != exp cfg epochs You are using {cmd_args.epochs}")
    EPOCHS = cmd_args.epochs

    data_folder = cmd_args.data_folder
    log_dir = get_logdir(cmd_args, exp_cfg,dist)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))
    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        cmd_args.train_replay_storage_dir,
        None,
        data_folder,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device_id,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        clip_cache_dir=cmd_args.clip_cache_dir,
        sequence_training=sequence_training,
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        full_episode=full_episode,
    )
    train_dataset, _ = get_dataset_func()
    t_end = time.time()
    if local_rank== 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    if int(getattr(mvt_cfg, "rot_ver", 0)) == 2:
        # BridgeVLA++ continuous 6D regression head: 6D rot + grip(2) + coll(2).
        mvt_cfg.feat_dim = 6 + 2 + 2
    else:
        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    if cmd_args.hidden_state_route_only and not mvt_cfg.hidden_state_enabled:
        raise ValueError(
            "--hidden_state_route_only requires hidden_state_enabled=True"
        )
    if (
        cmd_args.hidden_state_unfreeze_action_path
        and not cmd_args.hidden_state_route_only
    ):
        raise ValueError(
            "--hidden_state_unfreeze_action_path requires "
            "--hidden_state_route_only"
        )
    if (
        cmd_args.hidden_state_lora_discrete_action_path
        and not cmd_args.hidden_state_route_only
    ):
        raise ValueError(
            "--hidden_state_lora_discrete_action_path requires "
            "--hidden_state_route_only"
        )
    if (
        cmd_args.hidden_state_lora_discrete_action_path
        and cmd_args.hidden_state_unfreeze_action_path
    ):
        raise ValueError(
            "Discrete action LoRA and full action-path unfreezing are mutually exclusive"
        )
    if sequence_training and not mvt_cfg.hidden_state_enabled:
        raise ValueError(
            "hidden-state sequence training requires hidden_state_enabled=True"
        )
    filter_innovation_loss_weight = float(
        exp_cfg.rvt.hidden_state_filter_innovation_loss_weight
    )
    if filter_innovation_loss_weight > 0.0:
        if not mvt_cfg.hidden_state_filter_correction:
            raise ValueError(
                "hidden_state_filter_innovation_loss_weight requires "
                "mvt.hidden_state_filter_correction=True"
            )
        if not sequence_training:
            raise ValueError(
                "hidden_state_filter_innovation_loss_weight requires "
                "chronological hidden-state sequence training"
            )
    mvt_cfg.freeze()

    # for maintaining backward compatibility
    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    if cmd_args.hidden_state_route_only and cmd_args.init_checkpoint is None:
        raise ValueError(
            "--hidden_state_route_only requires --init_checkpoint so that the "
            "frozen BridgeVLA model starts from a trained checkpoint"
        )
    if cmd_args.init_checkpoint is not None and cmd_args.load_pretrain:
        raise ValueError(
            "Use either --init_checkpoint or --load_pretrain, not both"
        )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    )
    if cmd_args.init_checkpoint is not None:
        load_initial_checkpoint(backbone, cmd_args.init_checkpoint)
    if cmd_args.hidden_state_route_only:
        freeze_for_hidden_state_route(
            backbone,
            unfreeze_action_path=cmd_args.hidden_state_unfreeze_action_path,
            discrete_action_lora=cmd_args.hidden_state_lora_discrete_action_path,
            lora_rank=cmd_args.lora_rank,
            lora_alpha=cmd_args.lora_alpha,
        )

    backbone = backbone.to(local_rank)
    # Chronological sequence updates accumulate several forward graphs. The
    # frozen BatchNorm buffers must not be rebroadcast between those forwards.
    backbone = DDP(
        backbone,
        device_ids=[local_rank],
        find_unused_parameters=True,
        broadcast_buffers=not sequence_training,
    )

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        hidden_state_route_only=cmd_args.hidden_state_route_only,
        hidden_state_unfreeze_action_path=cmd_args.hidden_state_unfreeze_action_path,
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    freeze_names=["lm_head","embed_tokens"]
    if cmd_args.freeze_vision_tower:
        freeze_names.append("vision_tower")
        print("Freeze vision tower")

    for name, module in agent._network.named_modules():
        for freeze_name in freeze_names:
            if freeze_name in name:
                for param in module.parameters():
                    param.requires_grad = False
                break
    
    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    total_params_billion = total_params / 1e9  
    print(f'Total trainable parameters: {total_params_billion:.2f} billion')


    agent.build(training=True, device=device_id)
    start_epoch = 0
    end_epoch = EPOCHS

    if dist.get_rank() == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
    # Initialize Logging =>> W&B
    if dist.get_rank() == 0:
        wandb_mode = os.environ.get("WANDB_MODE", "")
        if wandb_mode == "disabled":
            wandb.init(mode="disabled")
        elif cmd_args.debug:
            wandb.init(
                entity="",
                project="",
                name=os.path.dirname(log_dir),
                mode="offline",
            )
        else:
            wandb.login(key="")
            wandb.init(entity="", project="", name=os.path.dirname(log_dir))

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        out = train(
            agent,
            train_dataset,
            TRAINING_ITERATIONS,
            epoch=i,
            rank=dist.get_rank(),
            sequence_training=sequence_training,
            sequence_bptt_length=bptt_length,
        )

        if dist.get_rank()==0 and (i %10==0 or i == end_epoch-1):
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1
        dist.barrier()

    dist.barrier()
    if dist.get_rank() == 0:
        print("[Finish]")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rlbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for matched training runs; unset preserves legacy behavior.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "outputs")),
        help="Root directory for training artifacts; defaults to <repo>/outputs.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument(
        "--train_replay_storage_dir", type=str, default=TRAIN_REPLAY_STORAGE_DIR
    )
    parser.add_argument("--clip_cache_dir", type=str, default=None)
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--hidden_state_route_only", action="store_true")
    parser.add_argument(
        "--hidden_state_unfreeze_action_path",
        action="store_true",
        help=(
            "With hidden_state_route_only, also train the continuous 6D "
            "action path (BridgeVLA++ rot_ver=2 feat_fc or the legacy "
            "continuous_rotation head) plus the convex-upsample output tails."
        ),
    )
    parser.add_argument(
        "--hidden_state_lora_discrete_action_path",
        action="store_true",
        help=(
            "With hidden_state_route_only and a discrete rot_ver=1 checkpoint, "
            "train zero-initialized LoRA deltas on the final x/y/z rotation "
            "and grip/collision projections while freezing the filter and base model."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--hidden_state_sequence_training",
        action="store_true",
        help="Train the hidden-state route on forward replay sequences.",
    )
    parser.add_argument(
        "--hidden_state_sequence_length",
        type=int,
        default=None,
        help="Legacy fixed-window length when full-episode mode is disabled.",
    )
    parser.add_argument(
        "--hidden_state_sequence_bptt_length",
        type=int,
        default=None,
        help="Maximum action-loss gradient span inside each episode.",
    )
    parser.add_argument(
        "--hidden_state_sequence_full_episode",
        action="store_true",
        help="Process complete replay episodes in chronological order.",
    )
    parser.add_argument(
        "--hidden_state_sequence_burn_in",
        type=int,
        default=None,
        help="Legacy fixed-window burn-in; must be zero in full-episode mode.",
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)
