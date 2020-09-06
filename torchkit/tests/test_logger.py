import torch

from torchkit.logger import Logger


# TODO: Make this test better.
class TestLogger:
    def test_log_scalar(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = Logger(log_dir)
        scalar = torch.randn(2, 4).mean()
        logger.log_scalar(scalar, 5, "train", "mean")

    def test_log_dict_scalars(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = Logger(log_dir)
        scalars = {
            'scalar1': torch.randn(2, 4).mean(),
            'scalar2': torch.randn(2, 4).mean(),
        }
        logger.log_dict_scalars(scalars, 5, "train")
