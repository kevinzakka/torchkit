import threading
from typing import Callable, Iterable, Tuple


# Adapted from: https://github.com/facebookresearch/pytorchvideo/blob/master/pytorchvideo/data/utils.py#L99  # noqa: E501
def threaded_func(
    func: Callable,
    args_iterable: Iterable[Tuple],
    multithreaded: bool,
) -> None:
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
