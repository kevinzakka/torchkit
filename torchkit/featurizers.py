import abc
import logging

import torch
import torch.nn as nn
import torchvision

from torchkit.utils.torch_utils import freeze_model

TensorType = torch.Tensor


class BaseFeaturizer(abc.ABC, nn.Module):
    """Abstract base class for featurizers.

    Subclasses must implement the `_build_featurizer` method.
    """

    def __init__(
        self,
        pretrained: bool = True,
        layers_train: str = "bn_only",
        bn_use_running_stats: bool = False,
    ):
        """Constructor.

        Args:
            pretrained: Whether to use Imagenet pretrained weights.
            layers_train: Controls which layers are trained. Can be one of
                `['all', 'frozen', 'bn_only']`.
            bn_use_running_stats: Set to `True` to disable batch statistics and
                use running mean and variance learned during training.
        """
        super().__init__()

        self._bn_use_running_stats = bn_use_running_stats
        self._layers_train = layers_train
        self._pretrained = pretrained

        self.model = self._build_featurizer()

        # Figure out batch norm related freezing parameters.
        if layers_train != "all":
            if layers_train == "frozen":
                logging.info("Freezing all featurizer layers.")
                bn_freeze_affine = True
            elif layers_train == "bn_only":
                logging.info("Freezing all featurizer layers except for "
                             "batch norm layers.")
                bn_freeze_affine = False
            else:
                raise ValueError(
                    "{} is not a valid layer selection strategy.".format(
                        layers_train
                    )
                )
            freeze_model(self.model, bn_freeze_affine, bn_use_running_stats)

        # Build param to module dict.
        self.param_to_module = {}
        for m in self.modules():
            for p in m.parameters(recurse=False):
                self.param_to_module[p] = type(m).__name__

    @abc.abstractmethod
    def _build_featurizer(self):
        """Build the featurizer architecture."""
        pass

    def forward(self, x: TensorType) -> TensorType:
        """Extract features from the video frames.

        Raises:
            ValueError: If the input tensor is not 2D (batch of images) or 3D
                (batch of sequence of images).
        """
        if x.ndim == 5:
            batch_size, t, c, h, w = x.shape
            x = x.view((batch_size * t, c, h, w))
            feats = self.model(x)
            _, c, h, w = feats.shape
            return feats.view((batch_size, t, c, h, w))
        elif x.ndim == 4:
            return self.model(x)
        else:
            raise ValueError("Only 2D or 3D data supported.")

    def train(self) -> None:
        """Sets the model in `train` mode."""
        self.training = True
        for m in self.model.modules():
            # Set everything that is NOT batchnorm to train.
            if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
            else:
                # For batch norm, we only want train mode if we were not using
                # running statistics.
                if self._bn_use_running_stats:
                    m.eval()
                else:
                    m.train()

    def eval(self) -> None:
        """Sets the model in `eval` mode."""
        self.training = False
        for m in self.model.modules():
            m.eval()

    @property
    def pretrained(self) -> bool:
        return self._pretrained


class ResNetFeaturizer(BaseFeaturizer):
    """A ResNet-based feature extractor."""

    RESNETS = ["resnet18", "resnet34", "resnet50", "resnet101"]
    RESNET_TO_CHANNELS = {
        "resnet18": 256,
        "resnet34": 256,
        "resnet50": 1024,
        "resnet101": 1024,
    }

    def __init__(
        self,
        model_type: str = "resnet18",
        out_layer_idx: int = 7,
        pretrained: bool = True,
        layers_train: str = "bn_only",
        bn_use_running_stats: bool = False,
    ):
        """Constructor.

        Args:
            model_type: The model type to use. Can be one of
                `['resnet18', 'resnet34', 'resnet50', 'resnet101']`.
            out_layer_idx: The index of the layer to use as output.
        """
        if model_type not in ResNetFeaturizer.RESNETS:
            raise ValueError(f"{model_type} is not a supported resnet model.")

        self._model_type = model_type
        self._out_layer_idx = out_layer_idx

        super().__init__(pretrained, layers_train, bn_use_running_stats)

    def _build_featurizer(self):
        resnet = getattr(torchvision.models, self._model_type)(
            pretrained=self._pretrained)
        layers_ = list(resnet.children())
        assert self._out_layer_idx < len(
            layers_
        ), "Output layer index exceeds total layers."
        layers_ = layers_[: self._out_layer_idx]
        return nn.Sequential(*layers_)


class InceptionFeaturizer(BaseFeaturizer):
    """An InceptionV3-based feature extractor."""

    LAYERS = [
        "Conv2d_1a_3x3",
        "Conv2d_2a_3x3",
        "Conv2d_2b_3x3",
        "MaxPool_3a_3x3",
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "MaxPool_5a_3x3",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
    ]

    LAYERS_TO_CHANNELS = {
        "Mixed_5d": 288,
    }

    def __init__(
        self,
        out_layer_name: str = "Mixed_5d",
        pretrained: bool = True,
        layers_train: str = "frozen",
        bn_use_running_stats: bool = False,
    ):
        """Constructor.

        Args:
            out_layer_name: Which layer of the inception model to use as the
                output.
        """
        assert (
            out_layer_name in InceptionFeaturizer.LAYERS
        ), f"{out_layer_name} is not supported."

        self._out_layer_name = out_layer_name

        super().__init__(pretrained, layers_train, bn_use_running_stats)

    def _build_featurizer(self):
        inception = torchvision.models.inception_v3(
            pretrained=self._pretrained, aux_logits=False, init_weights=False,
        )
        layers_, flag = [], False
        for name, module in inception.named_modules():
            if not name or "." in name:
                continue
            if self._out_layer_name in name:
                flag = True
            layers_.append(module)
            if flag:
                break
        return nn.Sequential(*layers_)
