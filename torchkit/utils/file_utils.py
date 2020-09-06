from typing import cast, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import os
import os.path as osp
import shutil

import numpy as np

from glob import glob
from PIL import Image


def mkdir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    if not osp.exists(path):
        os.makedirs(path)


def rm(path: str) -> None:
    """Remove a file or folder.

    Use with caution!

    Raises:
        IOError: If not a valid file or directory.
    """
    if osp.exists(path):
        if osp.isfile(path):
            os.remove(path)
        elif osp.isdir(path):
            shutil.rmtree(path)
        else:
            raise IOError("{} is not a file or directory.".format(path))


def get_subdirs(
    d: str,
    nonempty: bool = False,
    hidden: bool = False,
    sort: bool = True,
    basename: bool = False,
    sortfunc: Optional[Callable] = None,
) -> List[str]:
    """Return a list of subdirectories in a given directory.

    Args:
        d: The path to the directory.
        nonempty: Only return non-empty subdirs.
        hidden: Return hidden files as well.
        sort: Whether to sort by alphabetical order.
        basename: Only return the tail of the subdir paths.
        sortfunc : An optional sorting Callable to use if sort is set to `True`.
    """
    subdirs = [cast(str, f.path) for f in os.scandir(d) if f.is_dir()
               if not f.name.startswith('.')]
    if nonempty:
        subdirs = [f for f in subdirs if not is_folder_empty(f)]
    if sort:
        if sortfunc is None:
            subdirs.sort(key=lambda x: x.split("/")[-1])
        else:
            subdirs.sort(key=sortfunc)
    if basename:
        return [osp.basename(x) for x in subdirs]
    return subdirs


def get_files(
    d: str,
    pattern: str,
    sort: bool = False,
    lexicographical: bool = False,
) -> List[str]:
    """Return a list of files in a given directory.

    Args:
        d: The path to the directory.
        pattern: The wildcard to filter files with.
        sort: Whether to sort the returned list.
        lexicographical: If sort, use lexicographical order. Set to `False` for
            numerical ordering.
    """
    files = glob(osp.join(d, pattern))
    files = [f for f in files if osp.isfile(f)]
    if sort:
        if lexicographical:
            files.sort(key=lambda x: osp.basename(x))
        else:
            files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return files


def is_folder_empty(d: str) -> bool:
    """A folder is not empty if it contains >=1 non hidden files."""
    return len(glob(osp.join(d, "*"))) == 0


def load_jpeg(
    filename: str,
    resize: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Loads a JPEG image as a numpy array.

    Args:
        filename: The name of the image file.
        resize: The height and width of the loaded image. Set to `None` to keep
            original image dims.

    Returns:
        A numpy uint8 array.
    """
    img = Image.open(filename)
    if resize is not None:
        # PIL expects a (width, height) tuple.
        img = img.resize((resize[1], resize[0]))
    return np.asarray(img)


def write_image_to_jpeg(
    arr: np.ndarray,
    filename: str,
    jpeg_quality: int = 95,
) -> None:
    """Save a numpy uint8 image array as a JPEG image to disk.

    Args:
        arr: A numpy RGB uint8 image.
        filename: The filename of the saved image.
        jpeg_quality: The quality of the jpeg image.

    Raises:
        ValueError: If the numpy dtype is not uint8.
    """
    if arr.dtype is not np.dtype(np.uint8):
        raise ValueError("Image must be uint8.")
    jpeg_quality = np.clip(jpeg_quality, a_min=1, a_max=95)
    image = Image.fromarray(arr)
    image.save(filename, jpeg_quality=jpeg_quality)


def write_audio_to_binary(
    arr: np.ndarray,
    filename: str,
) -> None:
    """Save a numpy float64 audio array as a binary npy file.

    Args:
        arr: A numpy float64 array.
        filename: The filename of the saved audio file.

    Raises:
        ValueError: If the numpy dtype is not float64.
    """
    if arr.dtype is not np.dtype(np.float64):
        raise ValueError("Audio must be float64.")
    np.save(filename, arr, allow_pickle=True)


def write_timestamps_to_txt(
    arr: np.ndarray,
    filename: str,
) -> None:
    """Save a list of timestamps as a txt file.

    Args:
        arr: A numpy float64 array.
        filename: The filename of the saved txt file.
    """
    np.savetxt(filename, arr)


def flatten(list_of_list: Sequence[Any]) -> List[Any]:
    """Flattens a list of lists."""
    return [item for sublist in list_of_list for item in sublist]


def copy_folder(src: str, dst: str) -> None:
    """Copies a folder to a new location."""
    shutil.copytree(src, dst)


def move_folder(src: str, dst: str) -> None:
    """Moves a folder to a new location."""
    shutil.move(src, dst)


def copy_file(src: str, dst: str) -> None:
    """Copies a file to a new location."""
    shutil.copyfile(src, dst)


def move_file(src: str, dst: str) -> None:
    """Moves a file to a new location."""
    shutil.move(src, dst)


def write_json(filename: str, d: Mapping[Any, Any]) -> None:
    """Write a dict to a json file."""
    with open(filename, "w") as f:
        json.dump(d, f, indent=2)


def load_json(filename: str) -> Dict[Any, Any]:
    """Load a json file to a dict."""
    with open(filename, "r") as f:
        d = json.load(f)
    return d


def dict_from_list(x: Sequence[Any]) -> Dict[Any, Any]:
    """Creates a dictionary from a list.

    Args:
        x: A flat list.

    Returns:
        A dict where the keys are range(0, len(x)) and the values are the list
        contents.
    """
    return {k: v for k, v in zip(range(len(x)), x)}
