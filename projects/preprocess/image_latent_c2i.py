import os
import torch
import argparse
import datetime
import time
import torchvision
import logging
import math
import shutil
import accelerate
import torch
import torch.utils.checkpoint
import diffusers
import numpy as np
import torch.nn.functional as F
import einops
import json
import os.path as osp
import functools

from PIL import Image
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, AutoencoderDC
from nit.utils.misc_utils import instantiate_from_config
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder, default_loader
from torchvision.transforms.functional import hflip 
from safetensors.torch import save_file
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from nit.utils.model_utils import dc_ae_encode, sd_vae_encode

logger = get_logger(__name__, log_level="INFO")

# For Omegaconf Tuple
def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)


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



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

class ImagenetDataDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        x, y, p = self.dataset[i]
        return {"jpg": x, "cls": y, "path": p}

    def __len__(self):
        return len(self.dataset)

# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60
def get_train_sampler(global_batch_size, max_steps, resume_step):
    sample_indices = torch.arange(0, max_steps * global_batch_size,).to(torch.long)
    return sample_indices[resume_step * global_batch_size : ].tolist()


class ImagenetLoader():
    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config.dataloader.batch_size
        self.num_workers = data_config.dataloader.num_workers

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, data_config.dataset.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
        self.train_dataset = ImagenetDataDictWrapper(ImageFolder(data_config.dataset.data_dir, transform=transform))
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, global_batch_size, max_steps, resume_step):
        sampler = get_train_sampler(
            global_batch_size, max_steps, resume_step
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
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

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.tracker,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
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
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    
    if train_config.allow_tf32: # for A100
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup models
    weight_dtype = torch.float32
    if 'sd-vae' in model_config.vae:    
        sd_vae = AutoencoderKL.from_pretrained(model_config.vae).to(accelerator.device, dtype=weight_dtype)
        sd_vae.eval()
        sd_vae.requires_grad_(False)
        encode_func = functools.partial(sd_vae_encode, sd_vae)
    elif 'dc-ae' in model_config.vae:
        dc_ae = AutoencoderDC.from_pretrained(model_config.vae).to(accelerator.device, dtype=weight_dtype)
        dc_ae.eval()
        dc_ae.requires_grad_(False)
        encode_func = functools.partial(dc_ae_encode, dc_ae)
        
    

    # Setup Dataloader
    total_batch_size = (
        data_config.dataloader.batch_size * 
        accelerator.num_processes * 
        train_config.gradient_accumulation_steps
    )
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
    
    get_train_dataloader = ImagenetLoader(data_config)
    train_len = get_train_dataloader.train_len()
    train_config.max_train_steps = math.ceil(train_len / total_batch_size)
    train_dataloader = get_train_dataloader.train_dataloader(
        global_batch_size=total_batch_size, max_steps=train_config.max_train_steps, resume_step=global_steps, 
    )

    
    # Prepare Accelerate
    train_dataloader= accelerator.prepare(train_dataloader)    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {get_train_dataloader.train_len()/data_config.dataloader.batch_size}")
    logger.info(f"  Dataset Length = {get_train_dataloader.train_len()}")
    logger.info(f"  Instantaneous batch size per device = {data_config.dataloader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}")
    

    # Potentially load in the weights and states from a previous save
    if train_config.resume_from_checkpoint and resume_from_path != None:
        accelerator.print(f"Resuming from checkpoint {resume_from_path}")
        accelerator.load_state(resume_from_path)

        
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, train_config.max_train_steps), 
        disable = not accelerator.is_local_main_process
    )
    progress_bar.set_description("Optim Steps")
    progress_bar.update(global_steps)
    
    # prepare patch size and max sequence length
    # make directory
    os.makedirs(data_config.dataset.target_dir, exist_ok=True)
    def save_data(z, y, p):
        # p: 'datasets/imagenet1k/images/train/n01440764/n01440764_10026.JPEG'
        # target_folder: 'target_dir/n01440764'
        target_folder = os.path.join(data_config.dataset.target_dir, p.split('/')[-2])
        f_name = p.split('/')[-1].split('.')[0]
        os.makedirs(target_folder, exist_ok=True)
        single_data = dict(latent=z.contiguous(), label=y)
        save_file(single_data, os.path.join(target_folder, f'{f_name}.safetensors'))
    
    for step, batch in enumerate(train_dataloader, start=global_steps):
        for batch_key in batch.keys():
            if not isinstance(batch[batch_key], (list, str)):
                batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)
            x = batch['jpg']
            y = batch['cls']
            p = batch['path']
            if 'sd-vae' in model_config.vae:
                z_ori = encode_func(x)
                z_hflip = encode_func(hflip(x))
                z = torch.stack([z_ori, z_hflip], dim=1)
            elif 'dc-ae' in model_config.vae:
                z_ori = encode_func(x)
                z_hflip = encode_func(hflip(x))
                z = torch.stack([z_ori, z_hflip], dim=1)
            
            for i in range(len(p)):
                save_data(z[i], y[i], p[i])
            
        # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
        if accelerator.sync_gradients: 
            progress_bar.update(1)
            global_steps += 1
                
            if global_steps % train_config.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if train_config.checkpoints_total_limit is not None:
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
                                removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_steps}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
                accelerator.wait_for_everyone()
        
        if global_steps >= train_config.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

        
if __name__ == "__main__":
    args = parse_args()
    main(args)