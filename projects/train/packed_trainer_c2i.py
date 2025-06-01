#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import functools
import gc
import itertools
import json
import logging
import math
import os
import time
import random
import shutil
import importlib
import csv
import numpy as np
import os.path as osp
from pathlib import Path
from typing import List, Union
from packaging import version
from tqdm.auto import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate, Dataset
from torchvision import transforms
from torchvision.transforms import Normalize

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL

from timeit import default_timer as timer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from nit.schedulers.flow_matching.loss import FlowMatchingLoss
from nit.data.packed_c2i_data import C2ILoader
from nit.utils.misc_utils import (
    get_obj_from_str, get_dtype, instantiate_from_config
)
from nit.utils.train_utils import (
    update_ema, log_validation,
)
from nit.utils.gpu_memory_monitor import build_gpu_memory_monitor


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----General Training Arguments----
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="The config file for training.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="t2i_linear_attention",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    args = parser.parse_args()
    return args



def main(args):
    project_dir = args.project_dir
    config = OmegaConf.load(args.config)
    model_config = config.model 
    data_config = config.data
    train_config = config.training

    config_dir = osp.join(project_dir, 'configs')
    checkpoint_dir = osp.join(project_dir, 'checkpoints')
    logging_dir = osp.join(project_dir, 'logs')
    sample_dir = osp.join(project_dir, 'samples')

    if getattr(train_config, 'fsdp_config', None) != None:
        import functools
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            BackwardPrefetch, CPUOffload, ShardingStrategy, MixedPrecision, 
            StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
        )
        from accelerate.utils import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        fsdp_cfg = train_config.fsdp_config
        if train_config.mixed_precision == "fp16":
            dtype = torch.float16
        elif train_config.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32   
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy = {
                'FULL_SHARD': ShardingStrategy.FULL_SHARD,
                'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
                'NO_SHARD': ShardingStrategy.NO_SHARD,
                'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
                'HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
            }[fsdp_cfg.sharding_strategy],
            backward_prefetch = {
                'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
                'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
            }[fsdp_cfg.backward_prefetch],
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
            ),
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=fsdp_cfg.min_num_params
            ),
            cpu_offload = CPUOffload(offload_params=fsdp_cfg.cpu_offload),
            state_dict_type = {
                'FULL_STATE_DICT': StateDictType.FULL_STATE_DICT,
                'LOCAL_STATE_DICT': StateDictType.LOCAL_STATE_DICT,
                'SHARDED_STATE_DICT': StateDictType.SHARDED_STATE_DICT
            }[fsdp_cfg.state_dict_type],
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            limit_all_gathers = fsdp_cfg.limit_all_gathers,
            use_orig_params = fsdp_cfg.use_orig_params,
            sync_module_states = fsdp_cfg.sync_module_states,
            forward_prefetch = fsdp_cfg.forward_prefetch,
            activation_checkpointing = fsdp_cfg.activation_checkpointing,
        )
    else:
        fsdp_plugin = None

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.tracker,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
        fsdp_plugin=fsdp_plugin,
    )
    

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        OmegaConf.save(config=config, f=osp.join(config_dir, "config.yaml"))
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if train_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    total_batch_size = (
        data_config.dataloader.batch_size * 
        accelerator.num_processes * 
        train_config.gradient_accumulation_steps
    )
    if train_config.scale_lr:
        learning_rate = (
            train_config.learning_rate * 
            total_batch_size / train_config.learning_rate_base_batch_size
        )
    else:
        learning_rate = train_config.learning_rate
    
    
    # prepare model, dataloader, optimizer and scheduler
    model = instantiate_from_config(model_config.network).to(device=accelerator.device)
    model.train()
    if model_config.use_ema:
        ema_model = deepcopy(model)
        ema_model.train()
        ema_model.requires_grad_(False)
    # Handle mixed precision and device placement
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(model).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(model).dtype}. {low_precision_error_string}"
        )
    
    if accelerator.is_main_process:
        total_params = 0
        trainable_params = 0
        projector_params = 0
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            total_params += param.numel()  # Total number of elements in the parameter
            if param.requires_grad:          # Check if the parameter is trainable
                trainable_params += param.numel()
            if 'projector' in name:
                projector_params += param.numel()
        print(trainable_params, total_params, total_params-projector_params, trainable_params/total_params)
    
    # Optimizer creation
    target_optimizer = train_config.optimizer.get('target', 'torch.optim.AdamW')
    optimizer = get_obj_from_str(target_optimizer)(
        model.parameters(), lr=learning_rate, 
        **train_config.optimizer.get("params", dict())
    )

    # Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    global_steps = 0
    if train_config.resume_from_checkpoint:
        # normal read with safety check
        if train_config.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(train_config.resume_from_checkpoint)
        else:   # Get the most recent checkpoint
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = osp.join(checkpoint_dir, dirs[-1]) if len(dirs) > 0 else None

        if resume_from_path is None:
            logger.info(
                f"Checkpoint '{train_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_config.resume_from_checkpoint = None
        else:
            global_steps = int(resume_from_path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
            logger.info(f"Resuming from steps: {global_steps}")
    
    get_train_dataloader = C2ILoader(data_config)
    train_dataloader = get_train_dataloader.train_dataloader(
        rank=accelerator.process_index, world_size=accelerator.num_processes, 
        global_batch_size=total_batch_size, max_steps=train_config.max_train_steps, 
        resume_steps=global_steps, seed=args.seed
    )

    # LR Scheduler creation
    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=train_config.max_train_steps,
    )

    # Prepare for training
    # Prepare everything with our `accelerator`.
    if model_config.use_ema:
        ema_model, model, optimizer, lr_scheduler = accelerator.prepare(
            ema_model, model, optimizer, lr_scheduler
        )
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )

    # transport 
    loss_fn = FlowMatchingLoss(**OmegaConf.to_container(model_config.transport))
    if model_config.enc_type == 'radio':
        from nit.models.nvidia_radio.hubconf import radio_model
        encoder = radio_model(version=model_config.enc_dir, progress=True, support_packing=True)
        encoder.to(device=accelerator.device).eval()
        encoder.requires_grad_(False)
    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and getattr(train_config, 'tracker', 'wandb') != None:
        tracker_project_name = project_dir.split('/')[-1]
        # accelerator.init_trackers("mcga", config=config, init_kwargs=train_config.tracker_kwargs)
        accelerator.init_trackers(tracker_project_name, config=config, init_kwargs=train_config.tracker_kwargs)

    
    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor(logger)
    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

    # 15. Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {get_train_dataloader.train_len()/data_config.dataloader.batch_size}")
    logger.info(f"  Dataset Length = {get_train_dataloader.train_len()}")
    logger.info(f"  Instantaneous batch size per device = {data_config.dataloader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}")
    logger.info(
        "  GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    gpu_memory_monitor.reset_peak_stats()
    data_loading_times = []
    feat_enc_times = []
    
    # Potentially load in the weights and states from a previous save
    if train_config.resume_from_checkpoint and resume_from_path != None:
        accelerator.print(f"Resuming from checkpoint {resume_from_path}")
        accelerator.load_state(resume_from_path)

    progress_bar = tqdm(
        range(0, train_config.max_train_steps),
        initial=global_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_main_process,
    )

    
    for batch in train_dataloader:
        time_last_log = timer()
        data_load_start = timer()
        # load dataset from batch
        batch_image = [image.to(accelerator.device) for image in batch['image']]
        batch_label = batch['label'].squeeze(0).to(accelerator.device, torch.int)
        packed_latent = batch['latent'].squeeze(0).to(accelerator.device)
        noises = torch.randn_like(packed_latent)
        hw_list = batch['hw_list'].squeeze(0).to(torch.int)
        batch_size = hw_list.shape[0]
        
        dropout_prob = model_config.network.params.class_dropout_prob
        num_classes = model_config.network.params.num_classes
        if dropout_prob > 0:
            drop_ids = torch.rand(batch_label.shape[0], device=accelerator.device) < dropout_prob
            batch_label = torch.where(drop_ids, num_classes, batch_label)
        data_loading_times.append(timer() - data_load_start)
                
        feat_enc_start = timer()
        zs = []
        if model_config.enc_type == 'radio':
            with torch.no_grad(), accelerator.autocast():
                raw_images = [(image.unsqueeze(0)+1.0)/2.0 for image in batch_image]
                _, z = encoder.forward_pack(raw_images)
                zs.append(z)
        feat_enc_times.append(timer() - feat_enc_start)

        with accelerator.accumulate(model):
            # forward and calculate loss
            model_kwargs = dict(y=batch_label, hw_list=hw_list)
            fm_loss, proj_loss = loss_fn(model, batch_size, packed_latent, noises, model_kwargs, use_dir_loss=True, zs=zs)
            loss = fm_loss + model_config.proj_coeff * proj_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients and train_config.max_grad_norm > 0:
                all_norm = accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # 20.4.15. Make EMA update to target student model parameters
            if model_config.use_ema:
                update_ema(ema_model, model, model_config.ema_decay)
            global_steps += 1
            time_delta = timer() - time_last_log
            sps = batch_size / time_delta
            time_data_loading = np.mean(data_loading_times)
            time_feat_enc = np.mean(feat_enc_times)
            time_data_loading_pct = 100 * time_data_loading / time_delta
            time_feat_enc_pct = 100 * time_feat_enc / time_delta
            gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
            
            if global_steps % train_config.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and train_config.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(checkpoint_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= train_config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - train_config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = osp.join(checkpoint_dir, removing_checkpoint)
                            try:
                                shutil.rmtree(removing_checkpoint)
                            except:
                                pass
                save_path = osp.join(checkpoint_dir, f"checkpoint-{global_steps}")
                if accelerator.is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                

                if global_steps in train_config.checkpoint_list:
                    save_path = os.path.join(checkpoint_dir, f"save-checkpoint-{global_steps}")
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                time.sleep(10)
                torch.cuda.empty_cache()
            
                if global_steps % train_config.validation_steps == 0:
                    log_validation(model)
            logs = {
                # loss and lr
                "loss_denoising": fm_loss.detach().item(), 
                "loss_projector": proj_loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                # time and status
                "sps": sps,
                "data_loading(s)": time_data_loading,
                "data_loading(%)": time_data_loading_pct,
                "time_feat_enc(s)": time_feat_enc,
                "time_feat_enc(%)": time_feat_enc_pct,
                "memory_max_active(GiB)": gpu_mem_stats.max_active_gib,
                "memory_max_active(%)": gpu_mem_stats.max_active_pct,
                "memory_max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                "memory_max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                "memory_num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                "memory_num_ooms": gpu_mem_stats.num_ooms
            }
            if accelerator.sync_gradients and train_config.max_grad_norm > 0:
                logs.update({'grad_norm': all_norm.item()})
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            accelerator.log(logs, step=global_steps)
        if global_steps >= train_config.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

