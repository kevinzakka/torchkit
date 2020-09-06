from typing import Any, Optional, Sequence

import logging
import random
import numpy as np
import os.path as osp
import torch

from yacs.config import CfgNode
from torchkit.utils.file_utils import mkdir


def seed_rng(seed) -> None:
    """Seeds python, numpy, and torch RNGs."""
    logging.info(f'Seeding RNGs with seed {seed}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _update_config(
    config_dict: CfgNode,
    config_path: Optional[str] = None,
    override_list: Optional[Sequence[Any]] = None,
    serialize_path: Optional[str] = None,
) -> None:
    """Updates a config dict.

    This function first updates the base dict values with values from a config
    file, then optionally overrides these values with a provided list of args,
    and finally optionally serializes the updated contents back to a yaml file.

    Args:
        config: The base config dict.
        config_path: Path to an experiment yaml file containing config options
            we'd like to override.
        override_list: Additional command line args we'd like to override in the
            config file. This is a list containing strings in the even positions
            and any object in odd positions.
        serialize_path: Path to the directory where to serialize the config
            file. Set to `None` to prevent serialization.
    """
    # Override values given a yaml file.
    if config_path is not None:
        logging.info(f"Updating default config with {config_path}.")
        config_dict.merge_from_file(config_path)

    # Override values given a list of command line args.
    if override_list is not None:
        logging.info("Also overriding with command line args.")
        config_dict.merge_from_list(override_list)

    # Serialize the config file.
    if serialize_path is not None:
        with open(serialize_path, "w") as f:
            config_dict.dump(stream=f, default_flow_style=False)

    # Make config file immutable.
    config_dict.freeze()


def init_experiment(
    logdir: str,
    config: CfgNode,
    config_file: Optional[str] = None,
    override_list: Optional[Sequence[Any]] = None,
    transient: bool = False,
):
    """Initializes a training experiment.

    Instantiates the compute device (CPU, GPU), serializes the config file to
    the log directory, optionally updates the config variables with values from
    a provided yaml file and seeds the RNGs.

    Args:
        logdir: Path to the log directory.
        config: The module-wide config dict.
        config_file: Path to an experimental run yaml file containing config
            options we'd like to override.
        override_list: Additional command line args we'd like to override in the
            config file.
        transient: Set to `True` to make a transient session, i.e. a session
            where the logging and config params are not saved to disk. This is
            useful for debugging sessions.
    """
    # Init compute device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU {torch.cuda.get_device_name(device)}.")
    else:
        logging.info("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    # Create logdir and update config dict.
    if not transient:
        # If a yaml file already exists in the log directory, it means we're
        # resuming from a previous run. So we update the values of our config
        # file with the values in the yaml file.
        if osp.exists(osp.join(logdir, "config.yml")):
            logging.info(
                "Config yaml already exists in log dir. Resuming training."
            )
            _update_config(
                config, osp.join(logdir, "config.yml"), override_list)
        # If no yaml file exists in the log directory, it means we're starting a
        # new experiment. So we want to update our config file but also
        # serialize it to the log dir.
        else:
            # Create the log directory if it doesn't already exist.
            mkdir(logdir)

            _update_config(
                config,
                config_file,
                override_list,
                osp.join(logdir, "config.yml"),
            )
    else:
        logging.info("Transient model turned ON.")
        _update_config(config, config_file, override_list)

    # Seed RNGs.
    if config.SEED is not None:
        logging.info(f"Experiment seed: {config.SEED}.")
        seed_rng(config.SEED)
    else:
        logging.info("No RNG seed has been set for this experiment.")

    return config, device
