import time


class Stopwatch:
    """A simple timer for measuring elapsed time.

    Example usage::

        stopwatch = Stopwatch()
        some_func()
        print(f"some_func took: {stopwatch.elapsed()} seconds.")
        stopwatch.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def elapsed(self) -> float:
        """Return the elapsed time since the stopwatch was reset."""
        return time.time() - self.time

    def reset(self) -> None:
        """Reset the stopwatch, i.e. start the timer."""
        self.time = time.time()
