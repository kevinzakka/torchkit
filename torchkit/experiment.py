import random
import subprocess

import numpy as np
import torch


# Reference: https://stackoverflow.com/a/21901260
def git_revision_hash() -> str:
    """Return git revision hash as a string."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def seed_rngs(seed: int):
    """Seeds python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cudnn(deterministic: bool = False, benchmark: bool = True):
    """Set PyTorch-related CUDNN settings."""
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
