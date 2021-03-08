"""A PyTorch toolkit for research."""

import logging
import sys

from torchkit.logger import Logger  # noqa: F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
