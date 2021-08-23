"""Methods useful for running experiments."""

import functools
import pdb
import random
import subprocess
import uuid
from typing import Iterable, Iterator

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


# Reference: https://github.com/deepmind/jaxline/blob/master/jaxline/utils.py
def pdb_fallback(f):
    """Wraps f with a pdb callback."""

    @functools.wraps(f)
    def inner_wrapper(*args, **kwargs):
        """Main entry function."""
        try:
            return f(*args, **kwargs)
        # KeyboardInterrupt and SystemExit are not derived from BaseException,
        # hence not caught by the post-mortem.
        except Exception as e:
            pdb.post_mortem(e.__traceback__)
            raise

    return inner_wrapper


def string_from_kwargs(**kwargs) -> str:
    """Concatenate kwargs into an underscore-separated string.

    Used to generate an experiment name based on supplied config kwargs.
    """
    return "_".join([f"{k}={v}" for k, v in kwargs.items()])


def unique_id() -> str:
    """Generate a unique ID as specified in RFC 4122."""
    # See https://docs.python.org/3/library/uuid.html
    return str(uuid.uuid4())


# Reference: https://github.com/unixpickle/vq-voice-swap/blob/main/vq_voice_swap/util.py
def infinite_dataset(data_loader: Iterable) -> Iterator:
    """An infinite loop over a `torch.utils.DataLoader` Iterable."""
    while True:
        yield from data_loader
