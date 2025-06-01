import os
import datetime
import torchvision
import numpy as np
import torch
import ast
import json
import time


from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from accelerate.logging import get_logger
from safetensors.torch import load_file
from einops import rearrange
from functools import partial
from torchvision.transforms.functional import hflip

from .sampler_util import get_train_sampler, get_packed_batch_sampler

logger = get_logger(__name__, log_level="INFO")

PATCH_SIZE = 1

def resize_arr(pil_image, height, width):
    pil_image = pil_image.resize((width, height), resample=Image.Resampling.BICUBIC)

    return pil_image

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

def packed_collate_fn(batch):
    packed_latent = []
    label = []
    hw_list = []
    image = []
    for data in batch:
        C, H, W = data['latent'].shape
        latent = rearrange(
            data['latent'], 'c (h p1) (w p2) -> (h w) c p1 p2', p1=PATCH_SIZE, p2=PATCH_SIZE
        )
        packed_latent.append(latent)
        hw_list.append([H/PATCH_SIZE, W/PATCH_SIZE])
        label.append(data['label'])
        image.append(data['image'])
    packed_latent = torch.concat(packed_latent)
    label = torch.tensor(label)
    hw_list = torch.tensor(hw_list, dtype=torch.int32)
    return dict(image=image, latent=packed_latent, label=label, hw_list=hw_list)



class ImprovedPackedImageNetLatentDataset(Dataset):
    def __init__(self, packed_json, jsonl_dir, data_types, latent_dirs, image_dir):
        super().__init__()
        assert len(data_types) == len(latent_dirs)
        self.type_to_dir = dict()
        for i, data_type in enumerate(data_types):
            self.type_to_dir[data_type] = latent_dirs[i]
        self.image_dir = image_dir

        with open(packed_json, 'r') as fp:
            self.packed_dataset = json.load(fp)

        with open(jsonl_dir, 'r') as fp:
            self.dataset = [json.loads(line) for line in fp]
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_meta = self.dataset[index]
        
        data_item = dict()
        data_type = data_meta['type']
        latent_file = os.path.join(self.type_to_dir[data_type], data_meta['latent_file'])
        image_file = os.path.join(self.image_dir, data_meta['image_file'])

        data = load_file(latent_file)
        
        height = data_meta['latent_h'] * 16
        width = data_meta['latent_w'] * 16
        
        if data_type == 'native-resolution':
            preprocess = partial(resize_arr, height=height, width=width)
        else:
            assert height == width
            preprocess = partial(center_crop_arr, image_size=height)

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: preprocess(pil_image=pil_image)),
            transforms.Lambda(lambda pil_image: (pil_image, hflip(pil_image))),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        rand_idx = torch.randint(low=0, high=2, size=(1,)).item()
        data_item['image'] = transform(Image.open(image_file).convert("RGB"))[rand_idx]
        data_item['latent'] = data['latent'][rand_idx]
        data_item['label'] = data['label']
        return data_item


class C2ILoader():
    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config.dataloader.batch_size
        self.num_workers = data_config.dataloader.num_workers

        self.data_type = data_config.data_type
        
    
        if data_config.data_type == 'improved_pack':
            self.train_dataset = ImprovedPackedImageNetLatentDataset(
                **OmegaConf.to_container(data_config.dataset)
            )
        else:
            raise NotImplementedError
        
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, rank, world_size, global_batch_size, max_steps, resume_steps, seed):
        sampler = get_train_sampler(
            self.train_dataset, rank, world_size, global_batch_size, max_steps, resume_steps, seed
        )
        if self.data_type == 'improved_pack':
            batch_sampler = get_packed_batch_sampler(
                self.train_dataset.packed_dataset, rank, world_size, max_steps, resume_steps, seed
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=packed_collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
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
            drop_last=True
        )




