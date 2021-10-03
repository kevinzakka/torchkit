import os
from typing import Optional

import yaml
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict


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


def load_config(
    exp_dir: str,
    config: Optional[ConfigDict] = None,
) -> Optional[FrozenConfigDict]:
    """Load a config from an experiment directory.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (Optional[ConfigDict], optional): An optional config object to inplace
            update. If one isn't provided, a new config object is returned. Defaults to
            None.

    Returns:
        Optional[FrozenConfigDict]: The config file that was stored in the experiment
        directory. Frozen for safety.
    """
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # Inplace update the config if one is provided.
    if config is not None:
        config.update(cfg)
        return
    return FrozenConfigDict(cfg)


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
