import torch
import torch.nn as nn

TensorType = torch.Tensor


# TODO(kevin): This should be deprecated since it now exists as `torch.nn.SiLU`.
def swish(x: TensorType, inplace: bool = False) -> TensorType:
    if inplace:
        # Note the use of `sigmoid` as opposed to `sigmoid_`.
        # This is because `x` is required to compute the gradient during the
        # backward pass and calling sigmoid in place will modify its values.
        return x.mul_(x.sigmoid())
    return x * x.sigmoid()


class Swish(nn.Module):
    """Swish activation function [1].

    References:
        [1]: Searching for Activation Functions,
        https://arxiv.org/abs/1710.05941
    """

    def __init__(self, inplace: bool = False) -> None:
        """Constructor.

        Args:
            inplace: Perform the activation inplace.
        """
        super().__init__()

        self.inplace = inplace

    def forward(self, x: TensorType) -> TensorType:
        return swish(x, self.inplace)
