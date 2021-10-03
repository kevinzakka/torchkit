"""Methods useful for running experiments."""

import functools
import os
import pdb
import random
import subprocess
import uuid
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import yaml
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict


# Reference: https://stackoverflow.com/a/21901260
def git_revision_hash() -> str:
    """Return git revision hash as a string."""
    try:
        string = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        string = err.output
    return string.decode("ascii").strip()


def seed_rngs(seed: int, pytorch: bool = True) -> None:
    """Seed system RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)


def set_cudnn(deterministic: bool = False, benchmark: bool = True):
    """Set PyTorch-related CUDNN settings."""
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


# Reference: https://github.com/deepmind/jaxline/blob/master/jaxline/utils.py
def pdb_fallback(f):
    """Wraps f with a pdb callback."""

    @functools.wraps(f)
    def inner_wrapper(*args, **kwargs):
        """Main entry function."""
        try:
            return f(*args, **kwargs)
        # KeyboardInterrupt and SystemExit are not derived from BaseException,
        # hence not caught by the post-mortem.
        except Exception as e:
            pdb.post_mortem(e.__traceback__)
            raise

    return inner_wrapper


def string_from_kwargs(**kwargs) -> str:
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
    """An infinite loop over a `torch.utils.DataLoader` Iterable."""
    while True:
        yield from data_loader


# Reference: https://github.com/deepmind/jaxline/blob/master/jaxline/base_config.py
def validate_config(
    base_config: ConfigDict,
    config: ConfigDict,
    base_filename: str,
) -> None:
    """Ensures a config inherits from a base config.

    Args:
      base_config: The base config.
      config: The child config.
      base_filename: Can be one of 'pretraining' or 'rl'.

    Raises:
      ValueError: if the base config contains keys that are not present in config.
    """
    for key in base_config.keys():
        if key not in config:
            raise ValueError(
                f"Key {key} missing from config. This config is required to have "
                f"keys: {list(base_config.keys())}. See base_configs/{base_filename} "
                "for more details."
            )
        if isinstance(base_config[key], ConfigDict) and config[key] is not None:
            validate_config(base_config[key], config[key], base_filename)


def setup_experiment(
    exp_dir: str,
    config: ConfigDict,
    resume: bool = False,
) -> None:
    """Initializes an experiment."""
    #  If the experiment directory doesn't exist yet, creates it and dumps the config
    # dict as a yaml file and git hash as a text file. If it exists already, raises a
    # ValueError to prevent overwriting unless resume is set to True.
    if os.path.exists(exp_dir):
        if not resume:
            raise ValueError(
                "Experiment already exists. Run with --resume to continue."
            )
        load_config_from_dir(exp_dir, config)
    else:
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
            yaml.dump(ConfigDict.to_dict(config), fp)
        with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
            fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir: str,
    config: Optional[ConfigDict] = None,
) -> Optional[ConfigDict]:
    """Load experiment config."""
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # Inplace update the config if one is provided.
    if config is not None:
        config.update(cfg)
        return
    return ConfigDict(cfg)


def dump_config(exp_dir: str, config: ConfigDict) -> None:
    """Dump config to disk."""
    # Note: No need to explicitly delete the previous config file as "w" will overwrite
    # the file if it already exists.
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
    config: ConfigDict,
    update_dict: Optional[ConfigDict] = None,
    freeze: bool = False,
) -> Optional[ConfigDict]:
    """Makes a copy of a config and optionally updates its values."""
    # Using the ConfigDict constructor leaves the `FieldReferences` untouched unlike
    # `ConfigDict.copy_and_resolve_references`.
    new_config = ConfigDict(config)
    if update_dict is not None:
        new_config.update(update_dict)
    if freeze:
        return FrozenConfigDict(new_config)
    return new_config
