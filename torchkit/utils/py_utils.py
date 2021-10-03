import threading
import time
from typing import Callable, Iterable, Tuple


class Stopwatch:
    """A simple timer for measuring elapsed time."""

    def __init__(self) -> None:
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
) -> None:
    """Applies a func on a tuple of args with optional multithreading.

    Example usage::

        frames_paths = ["./images/img1.png", "./images/img2.png"]
        frames = [None for _ in range(len(frame_paths))]

        def get_image(image_index: int, image_path: str) -> None:
            frames[image_index] = np.asarray(Image.open(image_path))

        threaded_func(
            func=get_image,
            args_iterable=enumerate(frame_paths),
            multithreaded=True,
        )

        # Images are now stored in `frames` as numpy arrays.

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
