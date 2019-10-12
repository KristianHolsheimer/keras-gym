Probability Distributions
=========================

This is a collection of probability distributions that can be used as part of
a computation graph.

All methods are **differentiable**, including the ``sample`` method. This means
that they may be used in constructing loss functions that require quantities
like (cross)entropy or KL-divergence.

.. note::

    The :func:`CategoricalDist.sample
    <keras_gym.proba_dists.CategoricalDist.sample>` has not yet been
    implemented in a differentiable way, but it will use the approach outlined
    in `[ArXiv:1611.01144] <https://arxiv.org/abs/1611.01144>`_


Objects
-------

.. toctree::

    dists




