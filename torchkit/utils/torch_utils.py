"""Pytorch-related utils."""

import torch
from prettytable import PrettyTable


def freeze_model(
    model: torch.nn.Module,
    bn_freeze_affine: bool = False,
    bn_use_running_stats: bool = False,
) -> None:
    """Freeze PyTorch model weights.

    Args:
        model: The model to freeze, a subclass of `torch.nn.Module`.
        bn_freeze_affine: If True, freezes batch norm params gamma and beta.
        bn_use_running_stats: If True, switches from batch statistics to running
            mean and std. This is recommended for very small batch sizes.
    """
    for m in model.modules():
        if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad = False
            m.eval()
        else:
            if bn_freeze_affine:
                for p in m.parameters(recurse=False):
                    p.requires_grad = False
            else:
                for p in m.parameters(recurse=False):
                    p.requires_grad = True
            if bn_use_running_stats:
                m.eval()


def get_total_params(model: torch.nn.Module, trainable: bool = True,) -> int:
    """Get the total number of parameters in a PyTorch model [1].

    Example usage:

    ```python
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x):
                out = F.relu(self.fc1(x))
                return self.fc2(out)

        net = SimpleMLP()
        num_params = torch_utils.get_total_params(net)

        # prints the following:
        +------------+------------+
        |  Modules   | Parameters |
        +------------+------------+
        | fc1.weight |     48     |
        |  fc1.bias  |     16     |
        | fc2.weight |     32     |
        |  fc2.bias  |     2      |
        +------------+------------+
        Total Trainable Params: 98
    ```

    Args:
        model: The model, a subclass of `torch.nn.Module`.
        trainable: Only return trainable parameters.

    References:
        [1]: https://stackoverflow.com/a/62508086
    """
    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and trainable:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print("Total Trainable Params: {:,}".format(total_params))

    return total_params
