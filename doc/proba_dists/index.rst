Probability Distributions
=========================

This is a collection of probability distributions that can be used as part of
a computation graph.

All methods are **differentiable**, including the :func:`sample` method via the
*reparametrization trick* or variations thereof. This means that they may be used
in constructing loss functions that require quantities like (cross)entropy or
KL-divergence.


Objects
-------

.. toctree::

    dists




