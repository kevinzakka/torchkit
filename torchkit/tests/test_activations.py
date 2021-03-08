import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

from torchkit import activations


class TestActivations:
    @torch.no_grad()
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_swish(self, batch_size):
        c, h, w = 3, 32, 32
        x = torch.zeros(batch_size, c, h, w)
        x_np = x.numpy()
        actual = activations.swish(x)
        expected = x_np * (1.0 / (1 + np.exp(-x_np)))
        assert_allclose(actual, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_grad(self, batch_size):
        c, h, w = 3, 16, 16
        x = torch.zeros(
            batch_size, c, h, w, dtype=torch.double, requires_grad=True
        )
        assert gradcheck(activations.swish, x, eps=1e-6, atol=1e-4)
