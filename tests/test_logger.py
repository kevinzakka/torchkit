import numpy as np
import pytest
import torch

from torchkit.logger import Logger


@pytest.fixture
def init_logger(tmp_path):
    log_dir = tmp_path / "logs"
    logger = Logger(log_dir, force_write=False)
    return logger


class TestLogger:
    def test_error_on_init_existing(self, tmp_path, init_logger):
        _ = init_logger
        with pytest.raises(ValueError):
            _ = Logger(tmp_path / "logs")

    @pytest.mark.parametrize(
        "scalar", [torch.FloatTensor([5.0]), torch.randn([2, 2]).mean(), 5.0]
    )
    def test_log_scalar(self, init_logger, scalar):
        logger = init_logger
        logger.log_scalar(scalar, 0, "loss", "training")

    def test_log_scalar_notscalar(self, init_logger):
        logger = init_logger
        scalar = torch.FloatTensor([5.0, 5.0])
        with pytest.raises(ValueError):
            logger.log_scalar(scalar, 0, "loss", "training")

    @pytest.mark.parametrize(
        "image",
        [
            np.random.randint(0, 256, size=(224, 224, 3)),
            np.random.randint(0, 256, size=(2, 224, 224, 3)),
            torch.randint(0, 256, size=(3, 224, 224)),
            torch.randint(0, 256, size=(2, 3, 224, 224)),
        ],
    )
    def test_log_image(self, init_logger, image):
        logger = init_logger
        logger.log_image(image, 0, "image", "validation")

    @pytest.mark.parametrize(
        "image",
        [
            np.random.randint(0, 256, size=(3, 224, 224)),
            np.random.randint(0, 256, size=(2, 3, 224, 224)),
            torch.randint(0, 256, size=(224, 224, 3)),
            torch.randint(0, 256, size=(2, 224, 224, 3)),
        ],
    )
    def test_log_image_wrong_format(self, init_logger, image):
        logger = init_logger
        with pytest.raises(TypeError):
            logger.log_image(image, 0, "image", "validation")

    @pytest.mark.parametrize(
        "video",
        [
            np.random.randint(0, 256, size=(5, 224, 224, 3)),
            np.random.randint(0, 256, size=(4, 5, 224, 224, 3)),
            torch.randint(0, 256, size=(5, 3, 224, 224)),
            torch.randint(0, 256, size=(4, 5, 3, 224, 224)),
        ],
    )
    def test_log_video(self, init_logger, video):
        logger = init_logger
        logger.log_video(video, 0, "video", "training")

    def test_log_video_wrongdim(self, init_logger):
        logger = init_logger
        image = np.random.randint(0, 256, (224, 224, 3))
        with pytest.raises(ValueError):
            logger.log_video(image, 0, "video", "training")

    @pytest.mark.parametrize(
        "video",
        [
            np.random.randint(0, 256, size=(5, 3, 224, 224)),
            np.random.randint(0, 256, size=(4, 5, 3, 224, 224)),
            torch.randint(0, 256, size=(5, 224, 224, 3)),
            torch.randint(0, 256, size=(4, 5, 224, 224, 3)),
        ],
    )
    def test_log_video_wrongformat(self, init_logger, video):
        logger = init_logger
        with pytest.raises(TypeError):
            logger.log_video(video, 0, "video", "training")

    def test_learning_rate(self, init_logger):
        logger = init_logger
        param = torch.randn((32, 3), requires_grad=True)
        optim = torch.optim.Adam([param], lr=1e-3)
        logger.log_learning_rate(optim, 0, "training")

    def test_learning_rate_notoptim(self, init_logger):
        logger = init_logger
        param = torch.randn((32, 3), requires_grad=True)
        with pytest.raises(TypeError):
            logger.log_learning_rate(param, 0, "training")
