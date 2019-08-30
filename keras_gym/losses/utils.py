from ..base.errors import DistributionError
from ..utils import check_tensor

import tensorflow as tf
import tensorflow.keras.backend as K

DISTS = (
    'categorical',
    'beta',
)


def _check_dist(dist):
    if not isinstance(dist, str):
        raise DistributionError(
            "Expected type for 'dist'; expected str but got: {}"
            .format(type(dist)))
    if dist not in DISTS:
        raise DistributionError(
            "Unknown distribution: '{}', expected one of"
            .format(dist, ','.join(map("'{}'".format, DISTS))))


def cross_entropy(P, Z, dist, allow_surrogate=True):
    """

    This utility function computes the cross-entropy of an updateable policy
    :math:`\\pi_\\theta(a|s)` relative to a behavior policy :math:`b(a|s)`:

    .. math::

        H[b,\\pi_\\theta](s)\\ =\\ \\mathbb{E}_{a\\sim b(.|s)}
            \\left\\{-\\log \\pi_\\theta(a|s)\\right\\}

    This is intended to be used as part of a loss/objective function. As such,
    this may actually return a surrogate :math:`\\tilde{H}` instead. Such a
    surrogate is constructed in such a way that it satisfies:

    .. math::

        \\nabla_\\theta \\tilde{H}[b,\\pi_\\theta]
            \\ =\\ \\nabla_\\theta H[b,\\pi_\\theta]

    Parameters
    ----------
    P : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        A batch of distribution parameters :term:`P` of the behavior policy.
        For discrete action spaces, this is typically just a one-hot encoded
        version of a batch of taken actions :term:`A`.

    Z : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        Similar to :term:`P`, this is a batch of distribution parameters. In
        contrast to :term:`P`, however, :term:`Z` represents the primary
        updateable policy :math:`\\pi_\\theta(a|s)` instead of the
        behavior/target policy :math:`b(a|s)`.

    dist : str

        The distribution identifier, e.g. ``'categorical'`` or ``'beta'``.

    allow_surrogate : bool, optional

        Whether to allow the function to return a surrogate function instead of
        the true function. The surrogate is generally more numerically stable,
        but if your loss/objective is of the form :math:`J(\\theta) =
        H[b,\\pi_\\theta]\\,f(\\theta)` it breaks the surrogate condition, i.e.
        :math:`\\nabla_\\theta \\tilde{J}(\\theta) \\neq \\nabla_\\theta
        J(\\theta)`.

    """
    _check_dist(dist)
    check_tensor(Z, ndim=2)

    if dist == 'categorical':
        if allow_surrogate:
            pi = K.stop_gradient(K.softmax(Z, axis=1))
            Z_mean = K.expand_dims(tf.einsum('ij,ij->i', pi, Z), axis=1)
            logpi = Z - Z_mean  # surrogate, i.e. not truly log(pi)
        else:
            logpi = K.log(K.softmax(Z, axis=1))
        return tf.einsum('ij,ij->i', P, logpi)

    if dist == 'beta':
        pass

    raise NotImplementedError('cross_entropy({})'.format(dist))
