import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_checkpoint(ckpt_path, location='cpu', weights_only=None):
    map_location = torch.device(location) if isinstance(location, str) else location
    load_kwargs = {'map_location': map_location}
    if weights_only is not None:
        load_kwargs['weights_only'] = weights_only

    try:
        return torch.load(ckpt_path, **load_kwargs)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        load_kwargs.pop('weights_only', None)
        return torch.load(ckpt_path, **load_kwargs)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(load_checkpoint(ckpt_path, location=location, weights_only=True))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
