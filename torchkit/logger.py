from typing import cast, Any, Union, Mapping, Type

import numpy as np
import os.path as osp
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchkit.utils.torch_utils import UnNormalize

TensorType = torch.Tensor


class Logger:
    """A Tensorboard-based logger."""

    def __init__(self, log_dir: str, force_write: bool = False):
        """Constructor.

        Args:
            log_dir: The directory in which to store Tensorboard logs.
            force_write: Whether to force write to an already existing log dir.
                Set to `True` if resuming training.
        """
        self._log_dir = log_dir

        # Setup the summary writer.
        writer_dir = osp.join(self._log_dir, "train_logs")
        if osp.exists(writer_dir) and not force_write:
            raise ValueError(
                "You might be overwriting a directory that already "
                "has train_logs. Please provide a new experiment name "
                "or set --resume to True when launching train script."
            )
        self._writer = SummaryWriter(writer_dir)

    def close(self):
        self._writer.close()

    def log_scalar(
        self,
        scalar: Union[TensorType, float],
        global_step: int,
        prefix: str,
        name: str = "",
    ) -> None:
        """Log a scalar value.

        Args:
            scalar: A scalar `torch.Tensor` or float.
            global_step: The training iteration step.
            prefix: A prefix to prepend to the logged scalar.
            name: The name of the logged scalar.
        """
        if isinstance(scalar, torch.Tensor):
            if cast(torch.Tensor, scalar).ndim != 0:
                raise ValueError('Tensor must be scalar-valued.')
            scalar = cast(torch.Tensor, scalar).item()
        assert np.isscalar(scalar), "Not a scalar."
        msg = "/".join([prefix, name]) if name else prefix
        self._writer.add_scalar(msg, scalar, global_step)

    def log_dict_scalars(
        self,
        dict_scalars: Mapping[str, float],
        global_step: int,
        prefix: str,
    ) -> None:
        """Log a dictionary of scalars.

        Args:
            scalar: A scalar `torch.Tensor` or float.
            global_step: The training iteration step.
            prefix: A prefix to prepend to each logged scalar.
        """
        assert isinstance(dict_scalars, dict)
        for name, scalar in dict_scalars.items():
            self.log_scalar(scalar, global_step, prefix, name)

    def log_image(
        self,
        image: Union[TensorType, np.ndarray],
        global_step: int,
        prefix: str,
        name: str = "",
        dataformat: str = "CHW",
        denormalize: bool = False,
    ) -> None:
        """Log an image or batch of images.

        Args:
            image: A `torch.Tensor` or numpy array. Can be a single image
                or a batch of images. If a torch.Tensor is provided,
                the format should be `CHW`. If a numpy array is provided,
                the format should be `HWC`.
            global_step: The training iteration step.
            prefix: A prefix to prepend to the logged image(s).
            name: The name of the logged image(s).
            dataformat: How the image data is stored. The default value
                is `CHW` assuming the input is a `torch.Tensor`. If feeding
                numpy arrays, set it to `HWC`.
            denormalize: Whether to revert the ImageNet per-channel
                normalization that was applied in the dataloader.
        """
        assert image.ndim in [3, 4], (
            "Only a single image or batch of images is currently supported.")
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.ndim == 3:
                image = image.permute(2, 0, 1)
            else:
                image = image.permute(0, 3, 1, 2)
        if image.ndim == 4:
            image = torchvision.utils.make_grid(image, nrow=5)
        if denormalize:
            image = UnNormalize()(image)
        msg = "/".join([prefix, name]) if name else prefix
        self._writer.add_image(msg, image, global_step, dataformats=dataformat)

    def log_learning_rate(
        self,
        optimizer: Type[torch.optim.Optimizer],
        global_step: int,
    ) -> None:
        """Log the learning rate.

        Args:
            optimizer: An optimizer.
            global_step: The training iteration step.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        self.log_scalar(lr, global_step, "learning_rate")

    def log_loss(
        self,
        losses: Union[Mapping[str, float], float],
        global_step: int,
    ) -> None:
        """Log a loss.

        Args:
            losses: Either a scalar or a dict of scalars.
            global_step: The training iteration step.
        """
        if not isinstance(losses, dict):
            losses = {"": losses}
        self.log_dict_scalars(losses, global_step, "loss")

    def log_metric(
        self,
        metrics: Mapping[str, Any],
        global_step: int,
        metric_name: str,
    ) -> None:
        """Log a metric.

        Args:
            metrics: A dict containing the metric values. Possible keys can be
                'scalar' or 'image'. Values can be scalars, lists or dicts.
            global_step: The training iteration step.
            metric_name: The name of the metric.
        """
        assert isinstance(metrics, dict)
        for split, metric in metrics.items():
            if "scalar" in metric:
                if isinstance(metric["scalar"], dict):
                    for k, v in metric["scalar"].items():
                        self.log_scalar(
                            v,
                            global_step,
                            split,
                            "{}_{}".format(metric_name, k),
                        )
                else:
                    self.log_scalar(
                        metric["scalar"], global_step, split, metric_name
                    )
            if "image" in metric:
                img = metric["image"][0]
                self._writer.add_image(
                    "{}/image/{}".format(split, metric_name),
                    img_tensor=img,
                    global_step=global_step,
                    dataformats="HWC",
                )
