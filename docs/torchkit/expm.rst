Experiment Management
=====================

.. toctree::

    expm.config
    expm.experiment
    expm.checkpoint
    expm.logger

``torchkit`` uses `YACS <https://github.com/rbgirshick/yacs>`_
to organize a project and manage its various experimental configurations.
Quoting the `README <https://github.com/rbgirshick/yacs/blob/master/README.md>`_:

    "To use YACS for your project, you first create a project config file,
    typically called config.py or defaults.py. This file is the one-stop reference
    point for all configurable options. It should be very well documented and
    provide sensible defaults for all options."

``torchkit`` provides an example ``config.py`` you can use as a template for
your project. Here's an excerpt illustrating hyperparameters like the
seed, the batch size, and data augmentation transforms.

    .. code-block:: python

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
            # "color_jitter",
            "normalize",
        ]

        ...

        # ============================================== #
        # End of config file
        # ============================================== #
        CONFIG = _C

When setting up a new experiment run, the idea is to import this global ``CONFIG``
variable and only update it with experiment-specific changes. This is achieved by
creating a YAML configuration file (one for each run) with override values for
only the parameters you want to change in the default ``config.py`` file.

For illustration purposes, suppose in a new experiment, we want to decrease
the batch size and turn off data augmentation. To do this, we create the below
YAML file and save it to ``configs/example.yaml``.

    .. code-block:: yaml

        BATCH_SIZE: 2
        AUGMENTATION:
          TRAIN:
          - normalize

We're now ready to start a new experimental run. This is where ``torchkit`` steps
in and facilitates the process of updating the global ``CONFIG`` dict. The
``init_experiment`` method does 4 things:

1. It instantiates the compute device, using a GPU if it finds one.
2. It seeds all RNGs if a seed is set in ``config.py``.
3. It overrides the values of the ``CONFIG`` dict with the ones provided in the yaml file. Parameters not mentioned in the YAML file are left untouched.
4. It serializes a copy of the final updated config dict to a log directory. This allows the user to resume training with an exact copy of all the hyperparameters that were used for that run.

    .. code-block:: python

        # Suppose your project is called `example` and it contains a global
        # config file saved as `example/config.py`.

        import os.path as osp
        from example.config import CONFIG
        from torchkit.experiment import init_experiment

        experiment_name = 'example'
        config_path = 'configs/example.yaml'
        log_dir = osp.join(CONFIG.DIRS.LOG_DIR, experiment_name)

        config, device = init_experiment(log_dir, CONFIG, config_path)

        # The `config` (not CONFIG) dict now holds all the correct experiment-related
        # hyperparameters.

Sometimes, you may want to debug with some new parameters without saving the
config to disk and having to deal with re-writes or re-loading values. You
can specify that this a transient experiment session with:

    .. code-block:: python

        init_experiment(log_dir, CONFIG, config_path, transient=True)

Finally, sometimes, it may be more useful to modify a config value via a command
line argument rather than through a YAML file, for example when looping over
different hyperparameter values and performing a grid search. When this is the
case, you can use the ``override_list`` arg which gets executed after the global
config dict is updated with the experiment-specific YAML file.

    .. code-block:: python

        import os.path as osp
        from example.config import CONFIG
        from torchkit.experiment import init_experiment

        for weight_decay in [1e-5, 1e-4, 1e-3]:
            override_list = ["OPTIMIZER.WEIGHT_DECAY", weight_decay]
            log_dir = osp.join(CONFIG.DIRS.LOG_DIR, "test_tuning")
            config, device = init_experiment(
                log_dir, CONFIG, "configs/example.yaml", override_list)

