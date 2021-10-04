"""Methods useful for running experiments."""

import os
import uuid
from typing import Any

from ml_collections import config_dict

from torchkit.utils import dump_config, git_revision_hash, load_config

ConfigDict = config_dict.ConfigDict


def string_from_kwargs(**kwargs: Any) -> str:
    """Concatenate kwargs into an underscore-separated string.

    Used to generate an experiment name based on supplied config kwargs.
    """
    return "_".join([f"{k}={v}" for k, v in kwargs.items()])


def unique_id() -> str:
    """Generate a unique ID as specified in RFC 4122."""
    # See https://docs.python.org/3/library/uuid.html
    return str(uuid.uuid4())


def setup_experiment(
    exp_dir: str,
    config: ConfigDict,
    resume: bool = False,
) -> None:
    """Initializes an experiment.

    If the experiment directory doesn't exist yet, creates it and dumps the config
    dict as a yaml file and git hash as a text file.
    If it exists already, raises a ValueError to prevent overwriting unless resume is
    set to True.
    If it exists already and resume is set to True, inplace updates the config with the
    values in the saved yaml file.

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
        # Inplace-update the config using the values in the saved yaml file.
        load_config(exp_dir, config)
    else:
        # Dump config as a yaml file.
        dump_config(exp_dir, config)

        # Dump git hash as a text file.
        with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
            fp.write(git_revision_hash())
