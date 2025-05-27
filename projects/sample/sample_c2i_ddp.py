# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL, AutoencoderDC
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import functools
import argparse
from omegaconf import OmegaConf
from einops import rearrange
from nit.schedulers.flow_matching.samplers_c2i import euler_sampler, euler_maruyama_sampler
from nit.utils import init_from_ckpt
from nit.utils.misc_utils import instantiate_from_config
from nit.utils.model_utils import sd_vae_decode, dc_ae_decode


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup dtype
    dtype = torch.bfloat16

    # Load model:
    config = OmegaConf.load(args.config)
    model_config = config.model 
    
    if 'dc-ae' in model_config.vae_dir:
        dc_ae = AutoencoderDC.from_pretrained(model_config.vae_dir).to(device)
        if args.slice_vae:
            dc_ae.enable_slicing()
        if args.slice_vae:
            dc_ae.enable_slicing()
        spatial_downsample = 32
        decode_func = functools.partial(dc_ae_decode, dc_ae)
    elif 'sd-vae' in model_config.vae_dir:
        sd_vae = AutoencoderKL.from_pretrained(model_config.vae_dir).to(device)
        if args.slice_vae:
            sd_vae.enable_slicing()
        if args.slice_vae:
            sd_vae.enable_slicing()
        spatial_downsample = 8
        decode_func = functools.partial(sd_vae_decode, sd_vae)
    else: raise
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    # image resolution
    patch_size = int(model_config.network.params.patch_size)
    latent_h = int(args.height / spatial_downsample / patch_size)
    latent_w = int(args.width / spatial_downsample / patch_size)


    if args.interpolation != 'no':    
        model_config.network.params['custom_freqs'] = args.interpolation
        model_config.network.params['max_pe_len_h'] = latent_h
        model_config.network.params['max_pe_len_w'] = latent_w
        model_config.network.params['decouple'] = args.decouple
        model_config.network.params['ori_max_pe_len'] = int(args.ori_max_pe_len)
    
    model = instantiate_from_config(model_config.network).to(device=device, dtype=dtype)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=True)
    model.eval()  # important!
    
    if args.ag_config != None and args.ag_ckpt != None:
        ag_config = OmegaConf.load(args.ag_config)
        ag_model_config = ag_config.model 
        ag_model = instantiate_from_config(ag_model_config.network).to(device=device, dtype=dtype)
        init_from_ckpt(ag_model, checkpoint_dir=args.ag_ckpt, ignore_keys=None, verbose=True)
        ag_model.eval()  # important!
    else:
        ag_model = None
    
    
    # Create folder to save samples:
    train_iter = args.ckpt.split('/')[-2].split('-')[-1]
    folder_name = f"{train_iter}-{args.height}x{args.width}-{args.mode}-{args.num_steps}-" \
                  f"cfg-{args.cfg_scale}-low-{args.guidance_low}-high-{args.guidance_high}"
    if ag_model != None:
        sample_folder_dir = f"{args.sample_dir}/ag-{folder_name}"
    else:
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if args.interpolation != 'no':
        sample_folder_dir += f'-{args.interpolation}'
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for i in pbar:
        
        # Sample inputs:
        z = torch.randn(
            (n*latent_h*latent_w, model.in_channels, patch_size, patch_size), 
            device=device, dtype=dtype
        )
        y = torch.randint(0, args.num_classes, (n,), device=device)
        hw_list = torch.tensor([[latent_h, latent_w] for _ in range(n)], device=device, dtype=torch.int)
        seqlens = hw_list[:, 0] * hw_list[:, 1]
        cu_seqlens = torch.cat([
            torch.tensor([0], device=hw_list.device, dtype=torch.int32), 
            torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        ])
        
        can_pass = True
        for j in range(n):
            index = j * dist.get_world_size() + rank + total
            if not os.path.exists(f"{sample_folder_dir}/{index:06d}.png"):
                can_pass = False
        if can_pass:
            total += global_batch_size
            print('total: ', total)
            continue

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            ag_model=ag_model,
            latents=z,
            y=y,
            hw_list=hw_list,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError

            samples = rearrange(samples, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=latent_h, w=latent_w)
            samples = decode_func(samples)
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
            ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--config", type=str, default=None, help="Optional config to a SiT checkpoint.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="workdir/c2i/samples")
    parser.add_argument("--ag-config", type=str, default=None)
    parser.add_argument("--ag-ckpt", type=str, default=None)


    # model
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--slice_vae", action=argparse.BooleanOptionalAction, default=False) # only for ode
    
    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    parser.add_argument("--interpolation", type=str, choices=['no', 'linear', 'ntk-aware', 'ntk-by-parts', 'yarn', 'ntk-aware-pro1', 'ntk-aware-pro2', 'scale1', 'scale2'], default='no') # interpolation
    parser.add_argument("--ori-max-pe-len", default=None, type=int)
    parser.add_argument("--decouple", default=False, action="store_true") # interpolation
    
    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode
    

    args = parser.parse_args()
    main(args)
