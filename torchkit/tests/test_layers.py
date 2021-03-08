import pytest
import torch
from torch.testing import assert_allclose

from torchkit import layers


class TestLayers:
    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_conv2d_same_shape(self, kernel_size):
        b, c, h, w = 32, 64, 16, 16
        x = torch.randn(b, c, h, w)
        out = layers.conv2d(c, c * 2, kernel_size=kernel_size)(x)
        assert out.shape[2:] == x.shape[2:]

    def test_spatial_soft_argmax(self):
        b, c, h, w = 32, 64, 16, 16
        x = torch.zeros(b, c, h, w)
        true_max = torch.randint(0, 10, size=(b, c, 2))
        for i in range(b):
            for j in range(c):
                x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = 1000
        soft_max = layers.SpatialSoftArgmax(normalize=False)(x).reshape(b, c, 2)
        assert_allclose(true_max.float(), soft_max)

    def test_global_max_pool_1d(self):
        b, c, t = 4, 3, 16
        x = torch.randn(b, c, t)
        x_np = x.numpy()
        actual = layers.GlobalMaxPool1d()(x)
        expected = x_np.max(axis=(-1))
        assert_allclose(actual, expected)

    def test_global_max_pool_2d(self):
        b, c, h, w = 4, 3, 16, 16
        x = torch.randn(b, c, h, w)
        x_np = x.numpy()
        actual = layers.GlobalMaxPool2d()(x)
        expected = x_np.max(axis=(-1, -2))
        assert_allclose(actual, expected)

    def test_global_max_pool_3d(self):
        b, c, t, h, w = 4, 16, 5, 16, 16
        x = torch.randn(b, c, t, h, w)
        x_np = x.numpy()
        actual = layers.GlobalMaxPool3d()(x)
        expected = x_np.max(axis=(-1, -2, -3))
        assert_allclose(actual, expected)

    def test_global_average_pool_1d(self):
        b, c, t = 4, 3, 16
        x = torch.randn(b, c, t)
        x_np = x.numpy()
        actual = layers.GlobalAvgPool1d()(x)
        expected = x_np.mean(axis=(-1))
        assert_allclose(actual, expected)

    def test_global_average_pool_2d(self):
        b, c, h, w = 4, 3, 16, 16
        x = torch.randn(b, c, h, w)
        x_np = x.numpy()
        actual = layers.GlobalAvgPool2d()(x)
        expected = x_np.mean(axis=(-1, -2))
        assert_allclose(actual, expected)

    def test_global_average_pool_3d(self):
        b, c, t, h, w = 4, 16, 5, 16, 16
        x = torch.randn(b, c, t, h, w)
        x_np = x.numpy()
        actual = layers.GlobalAvgPool3d()(x)
        expected = x_np.mean(axis=(-1, -2, -3))
        assert_allclose(actual, expected)
