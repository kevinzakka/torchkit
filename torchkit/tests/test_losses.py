import numpy as np
import pytest
import torch

from torch.testing import assert_allclose
from torchkit import losses


def log_softmax(x):
    """A numerically stable log softmax implementation."""
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class TestLayers:
    @pytest.mark.parametrize("smooth_eps", [0, 0.5, 1])
    @pytest.mark.parametrize("K", [2, 100])
    def test_one_hot(self, smooth_eps, K):
        batch_size = 32
        y = torch.randint(K, (batch_size,))
        y_np = y.numpy()
        actual = losses.one_hot(y, K, smooth_eps)
        y_np_one_hot = np.eye(K)[y_np]
        expected = y_np_one_hot * (1 - smooth_eps) + (smooth_eps / (K - 1))
        assert_allclose(actual, expected)

    def test_one_hot_not_rank_one(self):
        with pytest.raises(AssertionError):
            losses.one_hot(torch.randint(5, (2, 2)), 5, 0)

    @pytest.mark.parametrize("smooth_eps", [-1, 2])
    def test_one_hot_eps_out_of_bounds(self, smooth_eps):
        with pytest.raises(AssertionError):
            losses.one_hot(torch.randint(5, (2, 2)), 5, smooth_eps)

    @pytest.mark.parametrize("smooth_eps", [0, 0.5, 1])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_cross_entropy(self, smooth_eps, reduction):
        batch_size = 2
        K = 5
        labels = torch.randint(K, (batch_size,))
        logits = torch.randn(batch_size, K)
        actual = losses.cross_entropy(logits, labels, smooth_eps, reduction)
        logits_np = logits.numpy()
        labels_np = labels.numpy()
        labels_np_one_hot = np.eye(K)[labels_np] * (1 - smooth_eps) \
            + (smooth_eps / (K - 1))
        # Compute log softmax of logits.
        log_probs_np = log_softmax(logits_np)
        loss_np = (-labels_np_one_hot * log_probs_np).sum(axis=-1)
        if reduction == "mean":
            expected = loss_np.mean()
        elif reduction == "sum":
            expected = loss_np.sum(axis=-1)
        else:  # none
            expected = loss_np
        assert_allclose(actual, expected)

    def test_cross_entropy_unsupported_reduction(self):
        K, batch_size = 5, 2
        with pytest.raises(AssertionError):
            labels = torch.randint(K, (batch_size,))
            logits = torch.randn(batch_size, K)
            losses.cross_entropy(logits, labels, reduction="average")

    def test_cross_entropy_labels_dim(self):
        K, batch_size = 5, 2
        with pytest.raises(AssertionError):
            labels = torch.randint(K, (batch_size, 2))
            logits = torch.randn(batch_size, K)
            losses.cross_entropy(logits, labels, reduction="average")

    @pytest.mark.parametrize("delta", [1, 2, 10])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_huber_loss(self, delta, reduction):
        batch_size = 2
        num_dims = 5
        target = torch.randn(batch_size, num_dims)
        input = torch.randn(batch_size, num_dims)
        target_np = target.numpy()
        input_np = input.numpy()
        actual = losses.huber_loss(input, target, delta, reduction)
        diff_abs_np = np.abs(target_np - input_np)
        diff_abs_np[diff_abs_np < delta] = 0.5 * diff_abs_np[
            diff_abs_np < delta] ** 2
        diff_abs_np[diff_abs_np >= delta] = delta * (
            diff_abs_np[diff_abs_np >= delta]  - 0.5 * delta)
        if reduction == "mean":
            expected = diff_abs_np.mean()
        elif reduction == "sum":
            expected = diff_abs_np.sum(axis=-1)
        else:  # none
            expected = diff_abs_np
        assert_allclose(actual, expected)
