.. automodule:: keras_gym.losses


Loss Functions
==============

This is a collection of custom `keras <https://keras.io/>`_-compatible loss
functions that are used throughout this package.

.. note:: These functions generally require the Tensorflow backend.


.. autosummary::
    :nosignatures:

    keras_gym.losses.SoftmaxPolicyLossWithLogits
    keras_gym.losses.masked_mse_loss



.. autoclass:: keras_gym.losses.SoftmaxPolicyLossWithLogits

    .. automethod:: __call__



.. autofunction:: keras_gym.losses.masked_mse_loss
