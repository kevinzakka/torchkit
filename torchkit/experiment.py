import logging
import random

import numpy as np
import torch


def seed_rng(
    seed: int, cudnn_deterministic: bool = True, cudnn_benchmark: bool = False,
) -> None:
    """Seeds python, numpy, pytorch and CUDA/cudNN RNGs [1].

    Args:
        seed: The seed to use.
        cudnn_deterministic: Enforce CUDA convolution determinism. The algorithm
            itself might not be deterministic so setting this to True ensures
            we make it repeatable.
        cudnn_benchmark: Set this to True to allow CUDA to find the best
            convolutional algorithm to use for the given parameters. When False,
            cuDNN will deterministically select the same algorithm at a possible
            cost in performance.

    References:
        [1]: https://pytorch.org/docs/stable/notes/randomness.html
    """
    logging.info(f"Seeding RNGs with seed {seed}.")
    # Seed for Python libraries.
    random.seed(seed)
    np.random.seed(seed)
    # Seed for PyTorch.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Seed for CUDA/cuDNN.
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
