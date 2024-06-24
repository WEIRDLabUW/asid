import random

import numpy as np
import omegaconf
import torch

import wandb


def hydra_to_dict(cfg):
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    return cfg_dict


def flatten_dict(cfg_dict):
    items = []
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_wandb(cfg, name, entity, project):

    run = wandb.init(
        name=name,
        entity=entity,
        project=project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
    )

    return run
