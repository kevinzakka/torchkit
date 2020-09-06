"""Default config variables.
"""

import os.path as osp

from yacs.config import CfgNode as CN

# ============================================== #
# Beginning of config file
# ============================================== #
_C = CN()

# ============================================== #
# Directories
# ============================================== #
_C.DIRS = CN()

_C.DIRS.DIR = osp.dirname(osp.realpath(__file__))
_C.DIRS.TEMP_DIR = osp.join(_C.DIRS.DIR, "tmp")
_C.DIRS.LOG_DIR = osp.join(_C.DIRS.DIR, "logs")
_C.DIRS.CKPT_DIR = osp.join(_C.DIRS.DIR, "checkpoints")

# ============================================== #
# Experiment params
# ============================================== #
# Seed for python, numpy and pytorch.
# Set this to `None` if you do not want to seed the results.
_C.SEED = 0

# Optimization level for mixed precision training using NVIDIA Apex.
# Can be one of [0, 1, 2].
_C.FP16_OPT = 0

_C.BATCH_SIZE = 4
_C.TRAIN_MAX_ITERS = 2000

# ============================================== #
# Data augmentation params
# ============================================== #
_C.AUGMENTATION = CN()

_C.IMAGE_SIZE = (224, 224)

_C.AUGMENTATION.TRAIN = [
    "global_resize",
    "horizontal_flip",
    # 'vertical_flip',
    # "color_jitter",
    # "rotate",
    "normalize",
]

_C.AUGMENTATION.EVAL = [
    "global_resize",
    "normalize",
]

# ============================================== #
# Evaluator params
# ============================================== #
_C.EVAL = CN()

# How many iterations of the validation data loader to run.
_C.EVAL.VAL_ITERS = 10

# ============================================== #
# Model params
# ============================================== #
_C.MODEL = CN()

## Feature extraction network
_C.MODEL.FEATURIZER = CN()

# Can be one of:
# 'resnet18', 'resnet34', 'resnet50', 'resnet101'
_C.MODEL.FEATURIZER.NETWORK_TYPE = "resnet18"

# Whether to use pretrained ImageNet weights.
_C.MODEL.FEATURIZER.PRETRAINED = True

# This variable controls how we want proceed with fine-tuning the base model.
# 'frozen': weights are fixed and batch_norm stats are also fixed.
# 'all': everything is trained and batch norm stats are updated.
# 'bn_only': only tune batch_norm variables and update batch norm stats.
_C.MODEL.FEATURIZER.TRAIN_BASE = "all"

# This variable controls how the batch norm layers apply normalization. If set
# to False, the mean and variance are calculated from the batch. If set to True,
# batch norm layers will normalize using the mean and variance of its moving
# (i.e. running) statistics, learned during training.
_C.MODEL.FEATURIZER.BN_USE_RUNNING_STATS = False

# These variables specify where to place the output head of the base model.
# Inception networks use a string.
# Resnet networks use an integer.
_C.MODEL.FEATURIZER.OUT_LAYER_NAME = "Mixed_5d"
_C.MODEL.FEATURIZER.OUT_LAYER_IDX = 7

# ============================================== #
# Loss params
# ============================================== #
_C.LOSS = CN()

# ============================================== #
# Optimizer params
# ============================================== #
_C.OPTIMIZER = CN()

# supported optimizers are: adam, sgd
_C.OPTIMIZER.TYPE = "adam"
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

# ============================================== #
# Logging params
# ============================================== #
_C.LOGGING = CN()

# number of steps between summary logging
_C.LOGGING.REPORT_INTERVAL = 100

# number of steps between eval logging
_C.LOGGING.EVAL_INTERVAL = 200

# ============================================== #
# Checkpointing params
# ============================================== #
_C.CHECKPOINT = CN()

# number of steps between consecutive checkpoints
_C.CHECKPOINT.SAVE_INTERVAL = 1000

# ============================================== #
# End of config file
# ============================================== #
CONFIG = _C
