import functools
from typing import Callable, TypeVar

CallableType = TypeVar("CallableType", bound=Callable)


# Reference: https://github.com/brentyi/fannypack/blob/master/fannypack/utils/_pdb_safety_net.py  # noqa: E501
def pdb_fallback(f: CallableType) -> CallableType:
    """Wraps a function in a pdb safety net for unexpected errors in a Python script.

    When called, pdb will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.

    Example usage::

        from torchkit.experiment import pdb_fallback

        @pdb_fallback
        def main():
            # A very interesting function that might fail because we did something
            # stupid.
            ...

        if __name__ == "__main__":
            main()
    """

    import signal
    import sys
    import traceback as tb

    import ipdb

    @functools.wraps(f)
    def inner_wrapper(*args, **kwargs):
        # Open pdb on Ctrl-C.
        def handler(sig, frame):
            ipdb.set_trace()

        signal.signal(signal.SIGINT, handler)

        # Open pdb when we encounter an uncaught exception.
        def excepthook(type_, value, traceback):
            tb.print_exception(type_, value, traceback, limit=100)
            ipdb.post_mortem(traceback)

        sys.excepthook = excepthook

        return f(*args, **kwargs)

    return inner_wrapper
