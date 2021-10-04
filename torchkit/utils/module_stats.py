import torch


# Reference: https://stackoverflow.com/a/62508086
def get_total_params(
    model: torch.nn.Module,
    trainable: bool = True,
    print_table: bool = False,
) -> int:
    """Get the total number of parameters in a PyTorch model.

    Example usage::

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()

                self.fc1 = nn.Linear(3, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = F.relu(self.fc1(x))
                return self.fc2(out)

        net = SimpleMLP()
        num_params = torch_utils.get_total_params(net, print_table=True)

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

    Args:
        model (torch.nn.Module): The pytorch model.
        trainable (bool, optional): Only consider trainable parameters. Defaults to
            True.
        print_table (bool, optional): Print the parameters in a pretty table. Defaults
            to False.

    Returns:
        int: Either all model parameters or only the trainable ones.
    """
    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and trainable:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    if print_table:
        print(table)
        print("Total Trainable Params: {:,}".format(total_params))

    return total_params
