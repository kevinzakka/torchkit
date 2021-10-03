from typing import Iterable, Iterator


# Reference: https://github.com/unixpickle/vq-voice-swap/blob/main/vq_voice_swap/util.py
def infinite_dataset(data_loader: Iterable) -> Iterator:
    """Create an infinite loop over a `torch.utils.DataLoader`.

    Args:
        data_loader (Iterable): A `torch.utils.DataLoader` object.

    Yields:
        Iterator: An iterator over the dataloader that repeats ad infinitum.
    """
    while True:
        yield from data_loader
