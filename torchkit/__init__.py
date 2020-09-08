"""A PyTorch toolkit for research.
"""

import sys
import logging

from torchkit.logger import Logger  # noqa: F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
