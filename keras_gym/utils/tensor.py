import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.linalg
from tensorflow.math import lbeta, digamma

from ..base.errors import TensorCheckError
from .misc import check_dist_id


__all__ = (
    'check_tensor',
    'cross_entropy',
    'diff_transform_matrix',
    'log_softmax_tf',
    'project_onto_actions_tf',
)


def check_tensor(tensor, ndim=None, ndim_min=None, dtype=None, int_shape=None, axis_size=None, axis=None):  # noqa: E501
    """

    This helper function is mostly for internal use. It is used to check a few
    common properties of a Tensor.

    Raises
    ------
    TensorCheckError

        If one of the checks fails, it raises a :class:`TensorCheckError`.

    """

    if not isinstance(tensor, tf.Tensor):
        raise TensorCheckError(
            "expected input to be a Tensor, got type: {}"
            .format(type(tensor)))

    check = ndim is not None
    if check and K.ndim(tensor) != ndim:
        raise TensorCheckError(
            "expected input with ndim equal to {}, got ndim: {}"
            .format(ndim, K.ndim(tensor)))

    check = ndim_min is not None
    if check and K.ndim(tensor) < ndim_min:
        raise TensorCheckError(
            "expected input with ndim at least {}, got ndim: {}"
            .format(ndim_min, K.ndim(tensor)))

    check = dtype is not None
    if check and tensor.dtype != dtype:
        raise TensorCheckError(
            "expected input with dtype {}, got dtype: {}"
            .format(dtype, tensor.dtype))

    check = int_shape is not None
    if check and K.int_shape(tensor) != int_shape:
        raise TensorCheckError(
            "expected input with shape {}, got shape: {}"
            .format(int_shape, K.int_shape(tensor)))

    check = axis_size is not None and axis is not None
    if check and K.int_shape(tensor)[axis] != axis_size:
        raise TensorCheckError(
            "expected input with size {} along axis {}, got shape: {}"
            .format(axis_size, axis, K.int_shape(tensor)))


def cross_entropy(P, Z, dist_id, allow_surrogate=True):
    """

    This utility function computes the *unaggregated* cross-entropy of an
    updateable policy :math:`\\pi_\\theta(a|s)` relative to a behavior policy
    :math:`b(a|s)`:

    .. math::

        H[b,\\pi_\\theta](s, a)\\ =\\ -b(a|s)\\,\\log\\pi_\\theta(a|s)

    To turn this into a loss, one needs to aggregate over the batch axis=0 (as
    usual) as well as aggregate/contract over axis=1 (i.e. over the actions
    :math:`a`)

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

    dist_id : str

        The policy distribution id, e.g. ``'categorical'`` or ``'beta'`` for
        a softmax policy or a Beta policy, respectively.

    allow_surrogate : bool, optional

        Whether to allow the function to return a surrogate function instead of
        the true function. The surrogate is generally more numerically stable,
        but if your loss/objective is of the form :math:`J(\\theta) =
        H[b,\\pi_\\theta]\\,f(\\theta)` it breaks the surrogate condition, i.e.
        :math:`\\nabla_\\theta \\tilde{J}(\\theta) \\neq \\nabla_\\theta
        J(\\theta)`.

    Returns
    -------
    cross_entropy : Tensor, shape: [batch_size, {num_actions,action_ndims}]

        A batch of *unaggregated* cross-entropy values
        :math:`H[b,\\pi_\\theta](s, a)`.

    """
    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z, ndim=2)
        check_tensor(P, ndim=2)
        check_tensor(Z, axis_size=K.int_shape(P)[0], axis=0)
        check_tensor(Z, axis_size=K.int_shape(P)[1], axis=1)

        if allow_surrogate:
            # Construct surrogate:
            # Let pi(a|s) = softmax(z(s,a)), then the surrogate for log(pi) is
            # logpi_surrogate = z(s,a) - sum_a' stop_gradient(pi(a'|s)) z(s,a')
            pi = K.stop_gradient(K.softmax(Z, axis=1))
            Z_mean = K.expand_dims(tf.einsum('ij,ij->i', pi, Z), axis=1)
            logpi = Z - Z_mean  # surrogate, i.e. not truly log(pi)

        else:
            logpi = K.log(K.softmax(Z, axis=1))

        return -P * logpi  # unaggregated, shape: [batch_size, num_actions]

    if dist_id == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution

        # expected input shapes: [batch_size, 2, actions_ndim]
        check_tensor(Z, ndim=3, axis_size=2, axis=1)
        check_tensor(P, ndim=3, axis_size=2, axis=1)

        # params of updateable policy: a_th = alpha(theta), b_th = beta(theta)
        a_th, b_th = tf.unstack(Z, axis=2)  # shapes: [batch, actions_ndim]

        # params of behavior policy: a = alpha, b = beta
        a, b = tf.unstack(K.max(1e-16, P), axis=2)  # clip for stability
        n = a + b  # shape: [batch, actions_ndim]
        p = a / n  # shape: [batch, actions_ndim]

        cross_entropy = K.switch(

            # check if n is large
            K.greater_equal(n, 1e4),

            # if so, use ordinary logpmf (digamma ~ log for large args)
            lbeta(Z) - (a_th - 1) * K.log(p) - (b_th - 1) * K.log(1 - p),

            # otherwise, use the exact cross-entropy
            lbeta(Z) - (a_th - 1) * digamma(a) - (b_th - 1) * digamma(b) \
            + (a_th + b_th - 2) * digamma(a + b),
        )
        return cross_entropy  # shape: [batch_size, actions_ndim]

    raise NotImplementedError('cross_entropy({})'.format(dist_id))


def diff_transform_matrix(num_frames, dtype='float32'):
    """
    A helper function that implements discrete differentiation for stacked
    state observations.

    Let's say we have a feature vector :math:`X` consisting of four stacked
    frames, i.e. the shape would be: ``[batch_size, height, width, 4]``.

    The corresponding diff-transform matrix with ``num_frames=4`` is a
    :math:`4\\times 4` matrix given by:

    .. math::

        M_\\text{diff}^{(4)}\\ =\\ \\begin{pmatrix}
            -1 &  0 &  0 & 0 \\\\
             3 &  1 &  0 & 0 \\\\
            -3 & -2 & -1 & 0 \\\\
             1 &  1 &  1 & 1
        \\end{pmatrix}

    such that the diff-transformed feature vector is readily computed as:

    .. math::

        X_\\text{diff}\\ =\\ X\\, M_\\text{diff}^{(4)}

    The diff-transformation preserves the shape, but it reorganizes the frames
    in such a way that they look more like canonical variables. You can think
    of :math:`X_\\text{diff}` as the stacked variables :math:`x`,
    :math:`\\dot{x}`, :math:`\\ddot{x}`, etc. (in reverse order). These
    represent the position, velocity, acceleration, etc. of pixels in a single
    frame.

    Parameters
    ----------
    num_frames : positive int

        The number of stacked frames in the original :math:`X`.

    dtype : keras dtype, optional

        The output data type.

    Returns
    -------
    M : 2d-Tensor, shape: [num_frames, num_frames]

        A square matrix that is intended to be multiplied from the left, e.g.
        ``X_diff = K.dot(X_orig, M)``, where we assume that the frames are
        stacked in ``axis=-1`` of ``X_orig``, in chronological order.

    """
    assert isinstance(num_frames, int) and num_frames >= 1
    s = np.diag(np.power(-1, np.arange(num_frames)))  # alternating sign
    m = s.dot(scipy.linalg.pascal(num_frames, kind='upper'))[::-1, ::-1]
    return K.constant(m, dtype=dtype)


def log_softmax_tf(Z, axis=-1):
    """
    Compute the log-softmax.

    **Note:** This is the *tensorflow* implementation.

    Parameters
    ----------
    Z : Tensor

        The input logits.

    axis : int, optional

        The axis along which to normalize, default is 0.

    Returns
    -------
    out : Tensor of same shape as input

        The entries may be interpreted as log-probabilities.

    """
    check_tensor(Z)
    Z = Z - K.mean(Z, axis=axis, keepdims=True)  # center before clipping
    Z = K.clip(Z, -30, 30)                       # avoid overflow before exp
    log_P = Z - K.log(K.sum(K.exp(Z), axis=axis, keepdims=True))
    return log_P


def project_onto_actions_tf(Y, A):
    """
    Project tensor onto specific actions taken: **tensorflow** implementation.

    **Note**: This only applies to discrete action spaces.

    Parameters
    ----------
    Y : 2d Tensor, shape: [batch_size, num_actions]

        The tensor to project down.

    A : 1d Tensor, shape: [batch_size]

        The batch of actions used to project.

    Returns
    -------
    Y_projected : 1d Tensor, shape: [batch_size]

        The tensor projected onto the actions taken.

    """
    # *note* Please let me know if there's a better way to do this.
    batch_size = tf.cast(K.shape(A)[0], tf.int64)
    idx = tf.range(batch_size, dtype=A.dtype)
    indices = tf.stack([idx, A], axis=1)
    Y_projected = tf.gather_nd(Y, indices)  # shape: [batch_size]
    return Y_projected
