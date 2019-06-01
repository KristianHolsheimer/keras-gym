Loss Functions
==============

This is a collection of custom `keras <https://keras.io/>`_-compatible loss
functions that are used throughout this package.

.. note:: These functions generally require the Tensorflow backend.


.. autosummary::
    :nosignatures:

    keras_gym.losses.ClippedSurrogateLoss
    keras_gym.losses.Huber
    keras_gym.losses.ProjectedSemiGradientLoss
    keras_gym.losses.RootMeanSquaredError
    keras_gym.losses.SoftmaxPolicyLossWithLogits


.. autoclass:: keras_gym.losses.ClippedSurrogateLoss
.. autoclass:: keras_gym.losses.Huber
.. autoclass:: keras_gym.losses.ProjectedSemiGradientLoss
.. autoclass:: keras_gym.losses.RootMeanSquaredError
.. autoclass:: keras_gym.losses.SoftmaxPolicyLossWithLogits
