import torch
from collections import OrderedDict
from copy import deepcopy
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def freeze_model(model, trainable_modules={}, verbose=False):
    logger.info("Start freeze")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if verbose:
            logger.info("freeze moduel: "+str(name))
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                if verbose:
                    logger.info("unfreeze moduel: "+str(name))
                break
    logger.info("End freeze")
    params_unfreeze = [p.numel() if p.requires_grad == True else 0 for n, p in model.named_parameters()]
    params_freeze = [p.numel() if p.requires_grad == False else 0 for n, p in model.named_parameters()]
    logger.info(f"Unfreeze Module Parameters: {sum(params_unfreeze) / 1e6} M")
    logger.info(f"Freeze Module Parameters: {sum(params_freeze) / 1e6} M")
    return 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def log_validation(model):
    pass