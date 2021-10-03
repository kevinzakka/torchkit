import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def one_hot(
    y: Tensor,
    K: int,
    smooth_eps: float = 0,
) -> Tensor:
    """One-hot encodes a tensor, with optional label smoothing.

    Args:
        y (Tensor): A tensor containing the ground-truth labels of shape `(N,)`, i.e.
            one label for each element in the batch.
        K (int): The number of classes.
        smooth_eps (float, optional): Label smoothing factor in `[0, 1]` range. Defaults
            to 0, which corresponds to no label smoothing.

    Returns:
        Tensor: The one-hot encoded tensor.
    """
    assert 0 <= smooth_eps <= 1
    assert y.ndim == 1, "Label tensor must be rank 1."
    y_hot = torch.eye(K)[y] * (1 - smooth_eps) + (smooth_eps / (K - 1))
    return y_hot.to(y.device)


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    smooth_eps: float = 0,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy loss with support for label smoothing.

    Args:
        logits (Tensor): A `FloatTensor` containing the raw logits, i.e. no softmax has
            been applied to the model output. The tensor should be of shape
            `(N, K)` where K is the number of classes.
        labels (Tensor): A rank-1 `LongTensor` containing the ground truth labels.
        smooth_eps (float, optional): The label smoothing factor in `[0, 1]` range.
            Defaults to 0.
        reduction (str, optional): The reduction strategy on the final loss tensor.
            Defaults to "mean".

    Returns:
        If reduction is `none`, a 2D Tensor.
        If reduction is `sum`, a 1D Tensor.
        If reduction is `mean`, a scalar 1D Tensor.
    """
    assert isinstance(logits, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(labels, (torch.LongTensor, torch.cuda.LongTensor))
    assert reduction in ["none", "mean", "sum"], "reduction method is not supported"

    # Ensure logits are not 1-hot encoded.
    assert labels.ndim == 1, "[!] Labels are NOT expected to be 1-hot encoded."

    if smooth_eps == 0:
        return F.cross_entropy(logits, labels, reduction=reduction)

    # One-hot encode targets.
    labels = one_hot(labels, logits.shape[1], smooth_eps)

    # Convert logits to log probabilities.
    log_probs = F.log_softmax(logits, dim=-1)

    loss = (-labels * log_probs).sum(dim=-1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    return loss.sum(dim=-1)


def huber_loss(
    input: Tensor,
    target: Tensor,
    delta: float,
    reduction: str = "mean",
) -> Tensor:
    """Huber loss with tunable margin, as defined in `1`_.

    Args:
        input (Tensor): A FloatTensor representing the model output.
        target (Tensor): A FloatTensor representing the target values.
        delta (float): Given the tensor difference `diff`, delta is the value at which
            we incur a quadratic penalty if `diff` is at least delta and a
            linear penalty otherwise.
        reduction (str, optional): The reduction strategy on the final loss tensor.
            Defaults to "mean".

    Returns:
        If reduction is `none`, a 2D Tensor.
        If reduction is `sum`, a 1D Tensor.
        If reduction is `mean`, a scalar 1D Tensor.

    .. _1: https://en.wikipedia.org/wiki/Huber_loss
    """
    assert isinstance(input, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(target, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert reduction in ["none", "mean", "sum"], "reduction method is not supported"

    diff = target - input
    diff_abs = torch.abs(diff)
    cond = diff_abs <= delta
    loss = torch.where(cond, 0.5 * diff ** 2, (delta * diff_abs) - (0.5 * delta ** 2))
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    return loss.sum(dim=-1)
