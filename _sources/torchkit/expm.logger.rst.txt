torchkit.logger
===============


The `Logger` class is a simple wrapper over Pytorch's ``SummaryWriter``. You can
use it to log losses, learning rates and metrics. For more fine-grained control,
you should use the ``log_scalar``, ``log_dict_scalars`` and ``log_image``
methods.

.. autoclass:: torchkit.logger.Logger
    :members:
    :member-order: groupwise
