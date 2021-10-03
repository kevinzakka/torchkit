"""Methods useful for running experiments."""

import functools
import os
import random
import subprocess
import uuid
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

import numpy as np
import torch
import yaml
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict
CallableType = TypeVar("CallableType", bound=Callable)


# Reference: https://stackoverflow.com/a/21901260
def git_revision_hash() -> str:
    """Return the git commit hash of the current directory.

    Note:
        Will return a `fatal: not a git repository` string if the command fails.
    """
    try:
        string = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        string = err.output
    return string.decode("ascii").strip()


# Alias.
git_commit_hash = git_revision_hash


def seed_rngs(seed: int, pytorch: bool = True) -> None:
    """Seed system RNGs.

    Args:
        seed (int): The desired seed.
        pytorch (bool, optional): Whether to seed the `torch` RNG as well. Defaults to
            True.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)


def set_cudnn(deterministic: bool = False, benchmark: bool = True) -> None:
    """Set PyTorch-related CUDNN settings.

    Args:
        deterministic (bool, optional): Make CUDA algorithms deterministic. Defaults to
            False.
        benchmark (bool, optional): Make CUDA arlgorithm selection deterministic.
            Defaults to True.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


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


def string_from_kwargs(**kwargs: Any) -> str:
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
    """Create an infinite loop over a `torch.utils.DataLoader`.

    Args:
        data_loader (Iterable): A `torch.utils.DataLoader` object.

    Yields:
        Iterator: An iterator over the dataloader that repeats ad infinitum.
    """
    while True:
        yield from data_loader


# Reference: https://github.com/deepmind/jaxline/blob/master/jaxline/base_config.py
def validate_config(base_config: ConfigDict, config: ConfigDict) -> None:
    """Ensures a config inherits from a base config.

    Args:
        base_config (ConfigDict): The base config.
        config (ConfigDict): The child config.

    Raises:
        ValueError: if the base config contains keys that are not present in config.
    """
    for key in base_config.keys():
        if key not in config:
            raise ValueError(
                f"Key {key} missing from config. This config is required to have "
                f"keys: {list(base_config.keys())}."
            )
        if isinstance(base_config[key], ConfigDict) and config[key] is not None:
            validate_config(base_config[key], config[key])


def dump_config(exp_dir: str, config: ConfigDict) -> None:
    """Dump a config to disk.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (ConfigDict): The config to dump.
    """
    # Note: No need to explicitly delete the previous config file as "w" will overwrite
    # the file if it already exists.
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
    config: ConfigDict,
    update_dict: Optional[ConfigDict] = None,
    freeze: bool = False,
) -> ConfigDict:
    """Makes a copy of a config and optionally updates its values.

    Args:
        config (ConfigDict): The config to copy.
        update_dict (Optional[ConfigDict], optional): A config that will optionally
            update the copy. Defaults to None.
        freeze (bool, optional): Whether to freeze the config after the copy. Defaults
            to False.

    Returns:
        ConfigDict: A copy of the config.
    """
    # Using the ConfigDict constructor leaves the `FieldReferences` untouched unlike
    # `ConfigDict.copy_and_resolve_references`.
    new_config = ConfigDict(config)
    if update_dict is not None:
        new_config.update(update_dict)
    if freeze:
        return FrozenConfigDict(new_config)
    return new_config


def setup_experiment(
    exp_dir: str,
    config: ConfigDict,
    resume: bool = False,
) -> None:
    """Initializes an experiment.

    If the experiment directory doesn't exist yet, creates it and dumps the config
    dict as a yaml file and git hash as a text file. If it exists already, raises a
    ValueError to prevent overwriting unless resume is set to True.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (ConfigDict): The config for the experiment.
        resume (bool, optional): Whether to resume from a previously created experiment.
            Defaults to False.

    Raises:
        ValueError: If the experiment directory exists already and resume is not set to
            True.
    """
    if os.path.exists(exp_dir):
        if not resume:
            raise ValueError(
                "Experiment already exists. Run with --resume to continue."
            )
        load_config_from_dir(exp_dir, config)
    else:
        os.makedirs(exp_dir)
        dump_config(exp_dir, config)
        with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
            fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir: str,
    config: Optional[ConfigDict] = None,
) -> Optional[ConfigDict]:
    """Load a config from an experiment directory.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (Optional[ConfigDict], optional): An optional config object to inplace
            update. If one isn't provided, a new config object is returned. Defaults to
            None.

    Returns:
        Optional[ConfigDict]: The config file that was stored in the experiment
        directory.
    """
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # Inplace update the config if one is provided.
    if config is not None:
        config.update(cfg)
        return
    return ConfigDict(cfg)
