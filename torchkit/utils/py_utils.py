import time
from typing import Any, Dict, Mapping, Sequence

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
