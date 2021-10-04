import os
import random

import numpy as np
import torch


def seed_rngs(seed: int, pytorch: bool = True) -> None:
    """Seed system RNGs.

    Args:
        seed (int): The desired seed.
        pytorch (bool, optional): Whether to seed the `torch` RNG as well. Defaults to
            True.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)


def set_cudnn(deterministic: bool = False, benchmark: bool = True) -> None:
    """Set PyTorch-related CUDNN settings.

    Args:
        deterministic (bool, optional): Make CUDA algorithms deterministic. Defaults to
            False.
        benchmark (bool, optional): Make CUDA arlgorithm selection deterministic.
            Defaults to True.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
