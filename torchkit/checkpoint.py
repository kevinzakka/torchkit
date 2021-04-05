import logging
import os
import os.path as osp
import signal
from glob import glob
from typing import Callable, List, Optional

import numpy as np
import torch


def get_files(
    d: str,
    pattern: str,
    sort: bool = False,
    lexicographical: bool = False,
    sortfunc: Optional[Callable] = None,
) -> List[str]:
    """Return a list of files in a given directory.

    Args:
        d: The path to the directory.
        pattern: The wildcard to filter files with.
        sort: Whether to sort the returned list.
        lexicographical: If sort, use lexicographical order. Set to `False` for
            numerical ordering.
        sortfunc : An optional sorting Callable to use if sort is set to `True`.
            Ignores the value of `lexicographical`.
    """
    if sortfunc is not None and not sort:
        raise ValueError("`sort` must be True when `sortfunc` is provided.")
    files = glob(osp.join(d, pattern))
    files = [f for f in files if osp.isfile(f)]
    if sort:
        if sortfunc is not None:
            files.sort(key=sortfunc)
        else:
            if lexicographical:
                files.sort(key=lambda x: osp.basename(x))
            else:
                files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return files


class Checkpoint:
    """Save and restore PyTorch objects implementing `state_dict` attributes.

    A `Checkpoint` object should be used in conjunction with a
    :class:`~torchkit.checkpoint.CheckpointManager`.

    Note: This is a re-implementation of `1`_.

    Example usage::

        import os
        from torchkit.checkpoint import Checkpoint

        checkpoint_dir = "/tmp/training_checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, "weights.ckpt")

        # Create a Checkpoint that will manage the model and optimizer.
        checkpoint = Checkpoint(model=model, optimizer=optimizer)
        checkpoint.save(checkpoint_path)
        status = checkpoint.restore(checkpoint_path)

    .. _1: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    """

    def __init__(self, **kwargs):
        """Constructor.

        Accepts keyword arguments whose values are objects that contain a
        `state_dict` attribute and thus can be serialized to disk.

        Args:
            kwargs: Keyword arguments are set as attributes of this object,
                and are saved with the checkpoint. Values must have a
                `state_dict` attribute.

        Raises:
            ValueError: If objects in `kwargs` do not have a `state_dict`
                attribute.
        """
        for k, v in sorted(kwargs.items()):
            if not getattr(v, "state_dict"):
                raise ValueError(f"{k} does not have a state_dict attribute.")
            setattr(self, k, v)

    def save(self, save_path: str) -> None:
        """Save a state to disk.

        Modified from brentyi/fannypack.

        Args:
            save_path: The name of the checkpoint to save.
        """
        # Ignore ctrl+c while saving.
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError:
            # Signal throws a ValueError if we're not in the main thread.
            orig_handler = None

        # Atomic save.
        save_dir = osp.dirname(save_path)
        tmp_path = osp.join(save_dir, f"tmp-{np.random.randint(1e10)}.ckpt")
        torch.save({k: v.state_dict() for k, v in self.__dict__.items()}, tmp_path)
        # Rename is POSIX-compliant and as such, is an atomic operation
        # according to the Python docs:
        # https://docs.python.org/3/library/os.html#os.rename
        os.rename(tmp_path, save_path)
        logging.info(f"Saved checkpoint at {save_path}.")

        # Restore SIGINT handler.
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)

    def restore(
        self,
        save_path: str,
        device: Optional[torch.device] = None,
    ) -> bool:
        """Restore a state from a saved checkpoint.

        Args:
            save_path: The filepath to the saved checkpoint.
            device: The device on which to restore the state.
        """
        try:
            state = torch.load(save_path, map_location=device)
            try:
                for name, state_dict in state.items():
                    getattr(self, name).load_state_dict(state_dict)
                logging.info(f"Successfully loaded model weights from {save_path}.")
                return True
            except Exception as e:
                # There was an issue loading the state which means either the
                # model definition and saved weights do not agree or they were
                # not saved in the first place.
                # Since this is a severe issue, we raise an error rather than
                # allowing the program to proceed.
                raise e
        except FileNotFoundError as e:
            logging.error(e)
            return False


class CheckpointManager:
    """Manages multiple checkpoints by keeping some and deleting unneeded ones.

    Note: This is a re-implementation of `2`_.

    Example usage::

        from torchkit.checkpoint import Checkpoint, CheckpointManager

        # Create a checkpoint manager instance.
        checkpoint_manager = checkpoint.CheckpointManager(
            checkpoint.Checkpoint(model=model, optimizer=optimizer),
            checkpoint_dir, device)

        # Restore last checkpoint if it exists.
        global_step = checkpoint_manager.restore_or_initialize()
        for global_step in range(1000):
            # forward pass + loss computation

            # Save a checkpoint every N iters.
            if not global_step % N:
                checkpoint_manager.save(global_step)

    .. _2: https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager/
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        directory: str,
        device: torch.device,
        max_to_keep: int = 10,
    ):
        """Constructor.

        Args:
            checkpoint: An instance of `Checkpoint`.
            directory: The directory in which checkpoints will be saved.
            device: The computing device on which to restore checkpoints.
            max_to_keep: The maximum number of checkpoints to keep.
                Amongst all saved checkpoints, checkpoints will be deleted
                oldest first, until `max_to_keep` remain.
        """
        assert max_to_keep > 0, "max_to_keep should be a positive integer."

        self.checkpoint = checkpoint
        self.directory = directory
        self.max_to_keep = max_to_keep
        self.device = device
        self.latest_checkpoint = None

        # Create checkpoint directory if it doesn't already exist.
        if not osp.exists(self.directory):
            os.makedirs(self.directory)

    def restore_or_initialize(self) -> int:
        """Restore items in checkpoint from the latest checkpoint file.

        Returns:
            The global iteration step. This is parsed from the latest checkpoint
            file if one is found, else 0 is returned.
        """
        ckpts = CheckpointManager.list_checkpoints(self.directory)
        if ckpts:
            last_ckpt = ckpts[-1]
            status = self.checkpoint.restore(last_ckpt, self.device)
            if not status:
                logging.info("Could not restore latest checkpoint file.")
                return 0
            self.latest_checkpoint = last_ckpt
            return int(osp.basename(last_ckpt).split(".")[0])
        return 0

    def save(self, global_step: int) -> None:
        """Create a new checkpoint.

        Args:
            global_step: The iteration number which will be used to name the
                checkpoint.
        """
        save_path = osp.join(self.directory, "{:016d}.ckpt".format(global_step))
        self.checkpoint.save(save_path)
        self.latest_checkpoint = save_path
        self._trim_checkpoints()

    def _trim_checkpoints(self):
        """Trim older checkpoints until `max_to_keep` remain."""
        # Get a list of checkpoints in reverse chronological order.
        ckpts = CheckpointManager.list_checkpoints(self.directory)[::-1]

        # Remove until `max_to_keep` remain.
        num_remove = len(ckpts) - self.max_to_keep
        while num_remove > 0:
            ckpt_name = ckpts.pop()
            os.remove(ckpt_name)
            num_remove -= 1

    @staticmethod
    def load_latest_checkpoint(
        checkpoint: Checkpoint,
        directory: str,
        device: torch.device,
    ) -> None:
        """Load the last saved checkpoint."""
        ckpts = CheckpointManager.list_checkpoints(directory)
        if ckpts:
            last_ckpt = ckpts[-1]
            checkpoint.restore(last_ckpt, device)
        else:
            raise ValueError(f"No checkpoints found in {directory}.")

    @staticmethod
    def list_checkpoints(directory: str) -> List[str]:
        """List all checkpoints in a checkpoint directory."""
        return get_files(
            directory,
            "*.ckpt",
            sort=True,
            sortfunc=lambda x: int(osp.splitext(osp.basename(x))[0]),
        )

    @staticmethod
    def load_specific_checkpoint(
        checkpoint: Checkpoint,
        checkpoint_filename: str,
        device: torch.device,
    ) -> None:
        """Load a specific checkpoint."""
        try:
            checkpoint.restore(checkpoint_filename, device)
        except Exception as e:
            raise e
