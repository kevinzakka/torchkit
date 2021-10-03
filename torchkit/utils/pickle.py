import logging
import os
import pickle
from typing import Any


def save_pickle(obj: Any, path: str, name: str) -> None:
    """Save a python object as a pickle file.

    Args:
        obj (Any): The object to save.
        path (str): Directory wherein to save file.
        name (str): Name of the pickle file.
    """
    filename = os.path.join(path, name)
    with open(filename, "wb") as fp:
        pickle.dump(obj, fp)
    logging.info(f"Successfully saved {filename}")


def load_pickle(path: str, name: str) -> Any:
    """Load a pickled file.

    Args:
        path (str): The directory where the pickle file is stored.
        name (str): The name of the pickle file.

    Returns:
        Any: The object in the pickle file.
    """
    filename = os.path.join(path, name)
    with open(filename, "rb") as fp:
        obj = pickle.load(fp)
    logging.info(f"Successfully loaded {filename}.")
    return obj
