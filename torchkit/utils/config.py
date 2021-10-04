import os
from typing import Optional, Union

import yaml
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict


# Reference: https://github.com/deepmind/jaxline/blob/master/jaxline/base_config.py
def validate_config(
    config: ConfigDict,
    base_config: ConfigDict,
    base_filename: str,
) -> None:
    """Ensures a config inherits from a base config.

    Args:
        config (ConfigDict): The child config.
        base_config (ConfigDict): The base config.
        base_filename (str): Path to python file containing base config definition.

    Raises:
        ValueError: if the base config contains keys that are not present in config.
    """
    for key in base_config.keys():
        if key not in config:
            raise ValueError(
                f"Key {key} missing from config. This config is required to have "
                f"keys: {list(base_config.keys())}. See {base_filename} for more "
                "details."
            )
        if isinstance(base_config[key], ConfigDict) and config[key] is not None:
            validate_config(config[key], base_config[key], base_filename)


def dump_config(exp_dir: str, config: Union[ConfigDict, FrozenConfigDict]) -> None:
    """Dump a config to disk.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (Union[ConfigDict, FrozenConfigDict]): The config to dump.
    """
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Note: No need to explicitly delete the previous config file as "w" will overwrite
    # the file if it already exists.
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(config.to_dict(), fp)


def load_config(
    exp_dir: str,
    config: Optional[ConfigDict] = None,
    freeze: bool = False,
) -> Optional[Union[ConfigDict, FrozenConfigDict]]:
    """Load a config from an experiment directory.

    Args:
        exp_dir (str): Path to the experiment directory.
        config (Optional[ConfigDict], optional): An optional config object to inplace
            update. If one isn't provided, a new config object is returned. Defaults to
            None.
        freeze (bool, optional): Whether to freeze the config. Defaults to False.

    Returns:
        Optional[Union[ConfigDict, FrozenConfigDict]]: The config file that was stored
        in the experiment directory.
    """
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # Inplace update the config if one is provided.
    if config is not None:
        config.update(cfg)
        return
    if freeze:
        return FrozenConfigDict(cfg)
    return cfg


def copy_config_and_replace(
    config: ConfigDict,
    update_dict: Optional[ConfigDict] = None,
    freeze: bool = False,
) -> Union[ConfigDict, FrozenConfigDict]:
    """Makes a copy of a config and optionally updates its values.

    Args:
        config (ConfigDict): The config to copy.
        update_dict (Optional[ConfigDict], optional): A config that will optionally
            update the copy. Defaults to None.
        freeze (bool, optional): Whether to freeze the config after the copy. Defaults
            to False.

    Returns:
        Union[ConfigDict, FrozenConfigDict]: A copy of the config.
    """
    # Using the ConfigDict constructor leaves the `FieldReferences` untouched unlike
    # `ConfigDict.copy_and_resolve_references`.
    new_config = ConfigDict(config)
    if update_dict is not None:
        new_config.update(update_dict)
    if freeze:
        return FrozenConfigDict(new_config)
    return new_config
