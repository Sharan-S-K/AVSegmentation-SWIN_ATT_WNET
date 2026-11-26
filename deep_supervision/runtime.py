from pathlib import Path
import random

import numpy as np
import torch

from models import AttentionWNetDeepSup


def setup(config, requested_device=None):
    seed = config.get('seed', 261)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get('cpu_threads', 4))
    device = torch.device(requested_device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return device


def build_model(config):
    settings = config['model_configs'][config['model_selection']]
    channels = 1 if config.get('channels') == 'green_only' else 3
    channels += 3 if config.get('use_lab_colorspace', False) else 0
    size = config['hardware_profiles'][config['hardware_profile']]['image_size']
    if size < 32 or size % 16:
        raise ValueError('Configured image_size must be at least 32 and divisible by 16')
    return AttentionWNetDeepSup(
        in_channels=channels, out_channels1=2,
        out_channels2=config['hyperparameters']['n_labels'],
        filters=settings['filter_num'], dropout=settings.get('dropout', 0))


def output_directory(config_path, config):
    output = Path(config['output_dir'])
    if not output.is_absolute():
        output = Path(config_path).resolve().parent / output
    output.mkdir(parents=True, exist_ok=True)
    return output
