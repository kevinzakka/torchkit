"""A PyTorch toolkit for research.
"""

import sys
import logging

from torchkit.logger import Logger  # noqa: F401
from torchkit.checkpoint import Checkpoint, CheckpointManager  # noqa: F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
