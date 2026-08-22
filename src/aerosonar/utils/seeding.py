"""Global random-number-generator seeding.

Training augmentation draws from the ``random`` module and SpecAugment from PyTorch's
generator, so seeding NumPy alone does not make a run reproducible. This module seeds
all three together.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed the Python, NumPy and PyTorch (CPU and CUDA) random number generators.

    Args:
        seed: Value applied to every generator.
        deterministic: If True, also restrict cuDNN to its deterministic algorithm
            set and disable benchmark autotuning. This costs throughput and is
            intended for reproducibility checks rather than for training.

    Returns:
        int: The seed that was applied, for logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
