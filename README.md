# torchkit

A set of PyTorch utilities that I use in day-to-day research.

* `config.py`: An example config file which should be used for sane default values.
* `checkpoint.py`: A simplified re-implementation of Tensorflow's checkpoint management tools.
* `experiment.py`: Boilerplate methods for initializing and setting up training runs.
* `layers.py`: Useful layers not available in pytorch for 1, 2 and 3D data.
* `losses.py`: Loss wrappers like cross entropy with label smoothing and huber loss.
* `featurizers.py`: Pretrained feature extractors with various types of freezing strategies.
* `utils`: A bunch of miscellaneous functions for file management and video processing.

**Table of Contents**

- [Logging](#logging)
- [Checkpointing](#checkpointing)
- [Featurizers](#featurizers)
- [Layers](#layers)
- [Experiment Management](#experiment-management)
- [Losses](#losses)
- [Utils](#utils)
  - [torch_utils](#torch_utils)
  - [file_utils](#file_utils)
  - [video_utils](#video_utils)

## Logging

The `Logger` class is a simple wrapper over Pytorch's `SummaryWriter`.

```python
from torchkit import Logger

# Log files to 'tmp' dir.
logger = Logger('./tmp')

global_step = 0
while True:
    model.train()

    # Iterate over dataloader here...
    input, target = next(data_loader)

    # Forward pass and compute loss
    output = model(input)
    loss_train = F.mse_loss(output, target)

    # Backwards pass and gradient step here.

    # Compute loss with model in eval mode.
    # Useful batch norm debugging strategy.
    model.eval()
    with torch.no_grad():
        loss_eval = F.mse_loss(model(input), target)

    losses = {'train': loss_train, 'eval': loss_eval}
    logger.log_loss(losses, global_step)

    # Or just log one loss.
    logger.log_loss(loss_train, global_step)

    # Log the learning rate.
    logger.log_learning_rate(optimizer, global_step)

    # Log the metric.
    accuracy = compute_acc(output, target)
    logger.log_metric({'scalar': accuracy})

    # Exit criterion here.
    global_step += 1
```

## Checkpointing

`Checkpoint` and `CheckpointManager` abstract away all the boilerplate
associated with saving and restoring PyTorch models, especially in the context
of multiple experiment runs.

```python
from torchkit import Checkpoint, CheckpointManager

# Create a checkpoint to track the model and optimizer states.
# A checkpoint object can be instantiated with any pytorch object that has a
# `state_dict` attribute.
checkpoint = Checkpoint(model=model, optimizer=optimizer)

# Wrap the checkpoint in a manager.
# The manager will manage multiple checkpoints by keeping some and deleting
# older ones.
# Here, we choose to only keep the last 20 model checkpoints.
checkpoint_manager = CheckpointManager(
    checkpoint, checkpoint_dir, device, max_to_keep=20)
```

The `CheckpointManager` fits nicely in the training loop paradigm, allowing us
to resume seamlessly from a previous checkpoint run.

```python
# Restore the last checkpoint if it exists. This will be 0 if just starting a
# training run.
global_step = checkpoint_manager.restore_or_initialize()
while True:
    # forward pass + loss computation

    # Save a checkpoint every N iters.
    if not global_step % SAVE_INTERVAL:
        checkpoint_manager.save(global_step)

    global_step += 1
```

## Featurizers

Featurizers are convenience wrappers around PyTorch ResNets and InceptionNets.
You specify whether you want to load ImageNet-pretrained weights and what layers
you want to train or freeze.

```python
from torchkit.featurizers import ResNetFeaturizer

# `layers_train` controls how we want to proceed with fine-tuning.
# 'frozen': Weights are fixed and batch_norm stats are also fixed.
# 'all': Everything is trained and batch norm stats are updated.
# 'bn_only': Only tune batch_norm variables and update batch norm stats.
#
# `bn_use_running_stats` controls how the batch norm layers apply
# normalization.
# If set to False, the mean and variance are calculated from the batch.
# If set to True, batch norm layers will normalize using the mean and variance
# of its moving (i.e. running) statistics, learned during training.
model = ResNetFeaturizer(
    model_type='resnet18',
    pretrained=True,
    layers_train='bn_only',
    bn_use_running_stats=False,
)
```

You can control which layer of the featurizer you want to use as output of the
model. For example, for a resnet18, `out_layer_idx=7` means the output feature
map will be of size `(*, 256, 14, 14)`.

## Layers

This module contains a bunch of common layers seen in papers but not currently
implemented in PyTorch:

* `conv2d` same, i.e. input shape equals output shape.
* `conv3d`: same, i.e. input shape equals output shape.
* `Flatten`
* `SpatialSoftArgmax`
* `GlobalMaxPool` (1D, 2D, 3D)
* `GlobalAvgPool` (1D, 2D, 3D)
* `CausalConv1d`

## Experiment Management

`torchkit` uses [yacs](https://github.com/rbgirshick/yacs) to setup a global
config file for the project. In the `config.py` file, the user is encouraged to
put sane default values for all the hyperparameters.

Here's an excerpt from the config file. You can see hyperparameters like the
seed, the batch size, augmentation strategies, etc.

```python
# ============================================== #
# Beginning of config file
# ============================================== #
_C = CN()

# ============================================== #
# Directories
# ============================================== #
_C.DIRS = CN()

_C.DIRS.DIR = osp.dirname(osp.realpath(__file__))
_C.DIRS.LOG_DIR = osp.join(_C.DIRS.DIR, "logs")
_C.DIRS.CKPT_DIR = osp.join(_C.DIRS.DIR, "checkpoints")

# ============================================== #
# Experiment params
# ============================================== #
# Seed for python, numpy and pytorch.
# Set this to `None` if you do not want to seed the results.
_C.SEED = 0

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

...

# ============================================== #
# End of config file
# ============================================== #
CONFIG = _C
```

When training a new model and setting up a new experiment run, you should import
the `CONFIG` variable and update it with experiment-specific changes. One way
of doing this is by having a separate yaml file with the *different* values
for the parameters and updating the `CONFIG` variable to use the yaml values
instead.

For illustration purposes, suppose in a new experiment, we want to decrease
the batch size and turn off data augmentation. We can create a yaml file with
these changes:

```yaml
# configs/example.yaml
BATCH_SIZE: 2
AUGMENTATION:
  TRAIN:
  - normalize
```

```python
import os.path as osp
from torchkit.config import CONFIG
from torchkit.experiment import init_experiment

experiment_name = 'example'
config_path = 'configs/example.yaml'

log_dir = osp.join(CONFIG.DIRS.LOG_DIR, experiment_name)

# This function does the following:
# 1. Instantiates the compute device, using a GPU if it finds one.
# 2. Seeds all RNGs.
# 3. Updates the default config values with the ones in the yaml file.
# 4. Serializes a copy of the final config dict to the log dir.
config, device = init_experiment(log_dir, CONFIG, config_path)

# The `config` (not CONFIG) dict now holds all the correct experiment-related
# hyperparameters.
```

Sometimes, you may want to debug with some new parameters without saving the
config to disk and having to deal with re-writes or re-loading values. You
can specify that this a transient experiment session with:

```python
config, device = init_experiment(log_dir, CONFIG, config_path, transient=True)
```

Finally, sometimes, it may be more useful to modify a config value via a command
line argument rather than through a yaml file, for example when looping over
different hyperparameter values and performing a grid search. When this is the
case, you can use the `override_list` arg which gets executed after the global
config dict is updated with the experiment-specific yaml file.

```python
import os.path as osp
from torchkit.config import CONFIG
from torchkit.experiment import init_experiment

for weight_decay in [1e-5, 1e-4, 1e-3]:
    override_list = ["OPTIMIZER.WEIGHT_DECAY", weight_decay]
    log_dir = osp.join(CONFIG.DIRS.LOG_DIR, "test_tuning")
    config, device = init_experiment(
        log_dir, CONFIG, "configs/example.yaml", override_list)
```

## Losses

We provide implementations of the following loss functions:

* `cross_entropy` with label smoothing
* `huber_loss`

## Utils

### torch_utils

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

### file_utils

A bunch of commonly used methods for dealing with datasets, files, etc. These
are mostly implemented to reduce the lines of code in a project.

* `mkdir`: Create a directory if it doesn't already exist.
* `rm`: Remove a file or a directory.
* `get_subdirs`: Return a list of subdirectories in a given directory.
    Optionally show hidden files, sort files, remove empty directories, etc.
* `get_files`: Return a list of files in a given directory.
* `load_jpeg`: Loads a JPEG image as a numpy array.
* `write_image_to_jpeg`: Save a numpy uint8 image array as a JPEG image to disk.
* `write_audio_to_binary`: Save a numpy float64 audio array as a binary npy file.
* `copy_folder`: Copy a folder to a new location.
* `move_folder`: Move a folder to a new location.
* `copy_file`: Copy a file to a new location.
* `move_file`: Move a file to a new location
* `write_json`: Write a dict to a json file.
* `load_json`: Load a json file to a dict.

### video_utils

* `video_fps`: Return a video's frame rate.
* `video_dimensions`: Return the (height, width) of a video.
* `video_to_frames`: Returns all frames from a video.
* `video_to_audio`: Returns all audio chunks from a video.
