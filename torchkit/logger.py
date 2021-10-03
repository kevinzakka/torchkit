import os.path as osp
from typing import Type, Union, cast

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

Tensor = torch.Tensor
ImageType = Union[Tensor, np.ndarray]


class Logger:
    """A Tensorboard-based logger."""

    def __init__(self, log_dir: str, force_write: bool = False) -> None:
        """Constructor.

        Args:
            log_dir: The directory in which to store Tensorboard logs.
            force_write: Whether to force write to an already existing log dir.
                Set to `True` if resuming training.
        """
        # Setup the summary writer.
        if osp.exists(log_dir) and not force_write:
            raise ValueError(
                "You might be overwriting a directory that already "
                "has train_logs. Please provide a new experiment name "
                "or set --resume to True when launching train script."
            )
        self._writer = SummaryWriter(log_dir)

    def close(self) -> None:
        self._writer.close()

    def flush(self) -> None:
        self._writer.flush()

    def log_scalar(
        self,
        scalar: Union[Tensor, float],
        global_step: int,
        name: str,
        prefix: str = "",
    ) -> None:
        """Log a scalar value.

        Args:
            scalar: A scalar `torch.Tensor` or float.
            global_step: The training iteration step.
            name: The name of the logged scalar.
            prefix: A prefix to prepend to the logged scalar.
        """
        if isinstance(scalar, torch.Tensor):
            if cast(torch.Tensor, scalar).ndim > 1:
                raise ValueError("Tensor must be scalar-valued.")
            if cast(torch.Tensor, scalar).ndim == 1:
                if cast(torch.Tensor, scalar).shape != torch.Size([1]):
                    raise ValueError("Tensor must be scalar-valued.")
            scalar = cast(torch.Tensor, scalar).item()
        assert np.isscalar(scalar), "Not a scalar."
        msg = "/".join([prefix, name]) if prefix else name
        self._writer.add_scalar(msg, scalar, global_step)

    def log_image(
        self,
        image: ImageType,
        global_step: int,
        name: str,
        prefix: str = "",
        nrow: int = 5,
    ) -> None:
        """Log an image or batch of images.

        Args:
            image: A numpy ndarray or a torch Tensor. If the image is 4D (i.e.
                batched), it will be converted to a 3D image using make_grid.
                The numpy array should be in channel-last format while the torch
                Tensor should be in channel-first format.
            global_step: The training iteration step.
            name: The name of the logged image(s).
            prefix: A prefix to prepend to the logged image(s).
            nrow: The number of images displayed in each row of the grid if the
                input image is 4D.
        """
        msg = "/".join([prefix, name]) if prefix else name
        assert image.ndim in [3, 4], "Must be an image or batch of images."
        if image.ndim == 4:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(0, 3, 1, 2)
            image = torchvision.utils.make_grid(image, nrow=nrow)
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1)
        self._writer.add_image(msg, image, global_step, dataformats="CHW")

    def log_video(
        self,
        video,
        global_step: int,
        name: str,
        prefix: str = "",
        fps: int = 4,
    ) -> None:
        """Log a sequence of images or a batch of sequence of images.

        Args:
            video: A torch Tensor or numpy ndarray. The numpy array should be in
                channel-last format while the torch Tensor should be in
                channel-first format. Should be either a single sequence of
                images of shape (T, CHW/HWC) or a batch of sequences of shape
                (B, T, CHW/HWC). The batch of sequences will get converted to
                one grid sequence of images.
            global_step: The training iteration step.
            name: The name of the logged video(s).
            prefix: A prefix to prepend to the logged video(s).
            fps: The frames per second.
        """
        msg = f"{prefix}/image/{name}"
        if video.ndim not in [4, 5]:
            raise ValueError("Must be a video or batch of videos.")
        if video.ndim == 4:
            if isinstance(video, np.ndarray):
                if video.shape[-1] != 3:
                    raise TypeError("Numpy array should have THWC format.")
                # (T, H, W, C) -> (T, C, H, W).
                video = torch.from_numpy(video).permute(0, 3, 1, 2)
            elif isinstance(video, torch.Tensor):
                if video.shape[1] != 3:
                    raise TypeError("Torch tensor should have TCHW format.")
            video = video.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W).
        else:
            if isinstance(video, np.ndarray):
                if video.shape[-1] != 3:
                    raise TypeError("Numpy array should have BTHWC format.")
                # (B, T, H, W, C) -> (B, T, C, H, W).
                video = torch.from_numpy(video).permute(0, 1, 4, 2, 3)
            elif isinstance(video, torch.Tensor):
                if video.shape[2] != 3:
                    raise TypeError("Torch tensor should have BTCHW format.")
        self._writer.add_video(msg, video, global_step, fps=fps)

    def log_learning_rate(
        self,
        optimizer: Type[torch.optim.Optimizer],
        global_step: int,
        prefix: str = "",
    ) -> None:
        """Log the learning rate.

        Args:
            optimizer: An optimizer.
            global_step: The training iteration step.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Optimizer must be an instance of torch.optim.Optimizer.")
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        self.log_scalar(lr, global_step, "learning_rate", prefix)
