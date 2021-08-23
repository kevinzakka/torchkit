import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


def dict_mean(
    dict_list: Sequence[Mapping[Any, Sequence[float]]],
) -> Dict[Any, float]:
    """Take the mean of a list of dicts.

    Raises:
        ValueError: If the dicts do not have the same keys.
    """
    # Ensure all dicts have the same keys.
    keys = dict_list[0].keys()
    for d in dict_list:
        if d.keys() != keys:
            raise ValueError("Dictionary keys must be identical.")

    # Combine list of dictionaries into one dictionary where each key's value is
    # a list containing the elements from all the subdictionaries.
    res = {k: [dict_list[i][k] for i in range(len(dict_list))] for k in keys}

    # Now take the average over the lists.
    return {k: np.mean(v) for k, v in res.items()}


class Stopwatch:
    """A simple timer for measuring elapsed time."""

    def __init__(self):
        self.reset()

    def elapsed(self) -> float:
        """Return the elapsed time since the stopwatch was reset."""
        return time.time() - self.time

    def reset(self) -> None:
        """Reset the stopwatch, i.e. start the timer."""
        self.time = time.time()


# Adapted from: https://github.com/facebookresearch/pytorchvideo/blob/master/pytorchvideo/data/utils.py#L99  # noqa: E501
def threaded_func(
    func: Callable,
    args_iterable: Iterable[Tuple],
    multithreaded: bool,
):
    """Applies a func on a tuple of args with optional multithreading.

    Args:
      func: The func to execute.
      args_iterable: An iterable of arg tuples to feed to func.
      multithreaded: Whether to parallelize the func across threads.
    """
    if multithreaded:
        threads = []
        for args in args_iterable:
            thread = threading.Thread(target=func, args=args)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
    else:
        for args in args_iterable:
            func(*args)
