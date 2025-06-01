import torch
import random
from typing import Optional, Union


def warp_pos_idx_from_grid(
    grid: torch.Tensor, 
    shift: Optional[int] = 0, 
    scale: Optional[str] = None, 
    max_len: Optional[Union[int, float]]=None
):
    '''
    grid: the 2-D positional index to be warped (B, 2, D)
    shift: the max shift value for the positional indices
    scale: the scale scheme for warping positional indices
    max_len: the max scale length
    '''
    grid[:, 0] = warp_pos_idx(grid[:, 0], shift, scale, max_len)
    grid[:, 1] = warp_pos_idx(grid[:, 1], shift, scale, max_len)
    return grid
    


def warp_pos_idx(
    pos_idx: torch.Tensor, 
    shift: Optional[int] = 0, 
    scale: Optional[str] = None, 
    max_len: Optional[Union[int, float]]=None
):
    '''
    pos_idx: the 1-D positional index to be warped (B, D)
    shift: the max shift value for the positional indices
    scale: the scale scheme for warping positional indices
    max_len: the max scale length
    '''
    if scale != None:
        assert isinstance(scale, str) and isinstance(max_len, (int, float))
        if scale.lower() == 'linear':
            pos_idx = max_len * (pos_idx / pos_idx.max())
        elif scale.lower() == 'sqrt':
            pos_idx = max_len * torch.sqrt(pos_idx / max_len)
        elif scale.lower() in ['sine', 'cosine', 'sin', 'cos']:
            pos_idx = max_len * torch.sin(pos_idx / max_len * (torch.pi/2))
        else:
            raise NotImplementedError('Only support linear, cosine, beta scale scheme for warping')
        
    pos_idx = pos_idx + random.randint(0, shift)
    
    return pos_idx

