from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import re
import os

from safetensors.torch import load_file


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    imgs = sorted(os.listdir(sample_dir), key=lambda x: int(x.split('.')[0]))
    print(len(imgs))
    assert len(imgs) >= num
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{imgs[i]}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None: 
    if checkpoint_dir.endswith(".safetensors"):
        model_state_dict=load_file(checkpoint_dir, device='cpu')
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]
    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=False)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")


def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sde-sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--ode-sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

# ode solvers:
# - Adaptive-step:
#   - dopri8 Runge-Kutta 7(8) of Dormand-Prince-Shampine
#   - dopri5 Runge-Kutta 4(5) of Dormand-Prince [default].
#   - bosh3 Runge-Kutta 2(3) of Bogacki-Shampine
#   - adaptive_heun Runge-Kutta 1(2)
# - Fixed-step:
#   - euler Euler method.
#   - midpoint Midpoint method.
#   - rk4 Fourth-order Runge-Kutta with 3/8 rule.
#   - explicit_adams Explicit Adams.
#   - implicit_adams Implicit Adams.