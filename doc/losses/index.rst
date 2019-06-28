Loss Functions
==============

This is a collection of custom `keras <https://keras.io/>`_-compatible loss
functions that are used throughout this package.

.. note:: These functions generally require the Tensorflow backend.


Value Losses
------------

These loss functions can be applied to learning a value function. Most of the
losses are actually already provided by `keras <https://keras.io/>`_. The
value-function losses included here are minor adaptations of the available
keras losses.


Policy Losses
-------------

These loss functions are used to implement policy-gradient (PG) methods. The
two main policy-gradient strategies provided by **keras-gym** are vanilla PG
and PPO-clipping, which are implemented using the loss functions:
:class:`SoftmaxPolicyLossWithLogits
<keras_gym.losses.SoftmaxPolicyLossWithLogits>` and
:class:`ClippedSurrogateLoss <keras_gym.losses.ClippedSurrogateLoss>`,
respectively.

Besides the PG-objective style loss functions, we also have some "loss"
functions that don't depend on the specific predictions made by the policy
object. They can be used as constraints or diagnostics (metrics). Most notably,
these include :class:`PolicyEntropy <keras_gym.losses.PolicyEntropy>` and
:class:`PolicyKLDivergence <keras_gym.losses.PolicyKLDivergence>`.


Objects
-------

.. toctree::

    value_based
    policy_based
