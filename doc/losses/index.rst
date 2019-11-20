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

The way policy losses are implemented is slightly different from value losses
due to their non-standard structure. A policy loss is implemented in a method
on :term:`updateable policy` objects (see below). If you need to implement a
custom policy loss, you can override this :func:`policy_loss_with_metrics`
method.

.. automethod:: keras_gym.core.base.BaseUpdateablePolicy.policy_loss_with_metrics


Objects
-------

.. toctree::

    value_based
