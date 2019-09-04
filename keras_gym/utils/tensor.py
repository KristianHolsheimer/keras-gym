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
    'entropy',
    'log_softmax_tf',
    'log_pi',
    'proba_ratio',
    'project_onto_actions_tf',
)


def check_tensor(
        tensor,
        ndim=None,
        ndim_min=None,
        dtype=None,
        same_dtype_as=None,
        same_shape_as=None,
        same_as=None,
        int_shape=None,
        axis_size=None,
        axis=None):  # noqa: E501

    """

    This helper function is mostly for internal use. It is used to check a few
    common properties of a Tensor.

    Parameters
    ----------
    ndim : int

        Check ``K.ndim(tensor)``.

    ndim_min : int

        Check if ``K.ndim(tensor)`` is at least ``ndim_min``.

    dtype : Tensor dtype

        Check ``tensor.dtype``.

    same_dtype_as : Tensor

        Check if dtypes match.

    same_shape_as : Tensor

        Check if shapes match.

    same_as : Tensor

        Check if both dtypes and shapes match.

    int_shape : tuple of ints

        Check ``K.int_shape(tensor)``.

    axis_size : int

        Check size along axis, where axis is specified by ``axis=...`` kwarg.

    axis : int

        The axis the check for size.

    Raises
    ------
    TensorCheckError

        If one of the checks fails, it raises a :class:`TensorCheckError`.

    """
    if not tf.is_tensor(tensor):
        raise TensorCheckError(
            "expected input to be a Tensor, got type: {}"
            .format(type(tensor)))

    if same_as is not None:
        assert tf.is_tensor(same_as), "not a tensor"
        same_dtype_as = same_as
        same_shape_as = same_as

    if same_dtype_as is not None:
        assert tf.is_tensor(same_dtype_as), "not a tensor"
        dtype = K.dtype(same_dtype_as)

    if same_shape_as is not None:
        assert tf.is_tensor(same_shape_as), "not a tensor"
        int_shape = K.int_shape(same_shape_as)

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
    if check and K.dtype(tensor) != dtype:
        raise TensorCheckError(
            "expected input with dtype {}, got dtype: {}"
            .format(dtype, K.dtype(tensor)))

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


def cross_entropy(Z1, Z2, dist_id, allow_surrogate=True):
    """

    This utility function computes the *unaggregated* cross-entropy of an
    updateable policy :math:`\\pi_{\\theta_2}(a|s)` relative to a another
    policy :math:`\\pi_{\\theta_1}(a|s)`:

    .. math::

        H[\\pi_{\\theta_1},\\pi_{\\theta_2}](s, a)\\ =\\
            -\\pi_{\\theta_1}(a|s)\\,\\log\\pi_{\\theta_2}(a|s)

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
    Z1 : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        A batch of distribution parameters. :term:`Z` contains the distribution
        parameters of the updateable policy :math:`\\pi_\\theta(a|s)`.

    Z2 : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        Same as ``Z1`` except that it comes from a different set of parameters
        :math:`\\theta_2\\neq\\theta_1` (in general).

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
        check_tensor(Z1, ndim=2)  # pi(a|s)
        check_tensor(Z2, ndim=2)  # b(a|s)
        check_tensor(Z2, axis_size=K.int_shape(Z1)[0], axis=0)
        check_tensor(Z2, axis_size=K.int_shape(Z1)[1], axis=1)

        # normalize logits
        pi1 = K.softmax(Z1, axis=1)
        pi2 = K.softmax(Z2, axis=1)

        if allow_surrogate:
            # Construct surrogate:
            # Let pi(a|s) = softmax(z(s,a)), then the surrogate for log(pi) is
            # logpi_surrogate = z(s,a) - sum_a' stop_gradient(pi(a'|s)) z(s,a')
            Z2_mean = K.expand_dims(
                tf.einsum('ij,ij->i', K.stop_gradient(pi2), Z2), axis=1)
            logpi2 = Z2 - Z2_mean  # surrogate, i.e. not truly log(pi)

        else:
            logpi2 = log_softmax_tf(Z2, axis=1)

        return -pi1 * logpi2  # unaggregated, shape: [batch_size, num_actions]

    if dist_id == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution

        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z1, ndim=3, axis_size=2, axis=2)
        check_tensor(Z2, ndim=3, axis_size=2, axis=2)

        # clip for numerical stability
        Z1 = K.maximum(1e-16, Z1)
        Z2 = K.maximum(1e-16, Z2)

        # params of behavior policy: a = alpha, b = beta
        p1, n1 = tf.unstack(Z1, axis=2)  # shapes: [batch, actions_ndim]
        a1 = n1 * p1
        b1 = n1 * (1 - p1)

        # params of updateable policy: a_th = alpha(theta), b_th = beta(theta)
        p2, n2 = tf.unstack(Z2, axis=2)  # shapes: [batch, actions_ndim]
        a2 = n2 * p2
        b2 = n2 * (1 - p2)
        ab2 = K.stack([a2, b2], -1)

        cross_entropy = (
            lbeta(ab2) - (a2 - 1) * digamma(a1) - (b2 - 1) * digamma(b1)
            + (a2 + b2 - 2) * digamma(a1 + b1))  # noqa: W503

        return cross_entropy  # shape: [batch_size, actions_ndim]

    raise NotImplementedError(
        "cross_entropy(dist_id='{}', ...)".format(dist_id))


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


def entropy(Z, dist_id):
    """

    This utility function computes the *unaggregated* entropy of an
    updateable policy :math:`\\pi_\\theta(a|s)`.

    .. math::

        H[\\pi_\\theta](s, a)\\ =\\ -\\pi_\\theta(a|s)\\,\\log\\pi_\\theta(a|s)

    To turn this into a loss, one needs to aggregate over the batch axis=0 (as
    usual) as well as aggregate/contract over axis=1 (i.e. over the actions
    :math:`a`)

    Parameters
    ----------
    Z : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        A batch of distribution parameters. :term:`Z` contains the distribution
        parameters of the updateable policy :math:`\\pi_\\theta(a|s)`.

    dist_id : str

        The policy distribution id, e.g. ``'categorical'`` or ``'beta'`` for
        a softmax policy or a Beta policy, respectively.

    Returns
    -------
    entropy : Tensor, shape: [batch_size, {num_actions,action_ndims}]

        A batch of *unaggregated* entropy values :math:`H[\\pi_\\theta](s, a)`.

    """
    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z, ndim=2)

        # action propensities
        pi = K.softmax(Z, axis=1)
        logpi = log_softmax_tf(Z, axis=1)

        return -pi * logpi  # unaggregated, shape: [batch_size, num_actions]

    if dist_id == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution

        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z, ndim=3, axis_size=2, axis=2)

        # clip for numerical stability
        Z = K.maximum(1e-16, Z)

        # dist params of policy: a = alpha(theta), b = beta(theta)
        p, n = tf.unstack(Z, axis=2)  # shapes: [batch, actions_ndim]
        a = n * p
        b = n * (1 - p)
        ab = tf.stack([a, b], -1)

        entropy = (
            lbeta(ab) - (a - 1) * digamma(a) - (b - 1) * digamma(b)
            + (a + b - 2) * digamma(a + b))  # noqa: W503

        return entropy  # shape: [batch_size, actions_ndim]

    raise NotImplementedError("entropy(dist_id='{}', ...)".format(dist_id))


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


def log_pi(P, Z, dist_id, allow_surrogate=True):
    """

    This utility function computes the *unaggregated* log-pdf of an updateable
    policy :math:`\\log\\pi_\\theta(a|s)` evaluated on action(s) sampled from a
    behavior policy :math:`a\\sim b(.|s)`:

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
    log_pi : Tensor, shape: [batch_size, {num_actions,action_ndims}]

        A batch of *unaggregated* log-pdf :math:`\\log\\pi_\\thet(a|s)`.

    """
    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z, ndim=2)     # Z = params of pi(a|s)
        P.set_shape(Z.get_shape())  # P = params of b(a|s)

        if allow_surrogate:
            # Construct surrogate:
            # Let pi(a|s) = softmax(z(s,a)), then the surrogate for log(pi) is
            # logpi_surrogate = z(s,a) - sum_a' stop_gradient(pi(a'|s)) z(s,a')
            pi = K.stop_gradient(K.softmax(Z, axis=1))
            Z_mean = K.expand_dims(tf.einsum('ij,ij->i', pi, Z), axis=1)
            logpi = Z - Z_mean  # surrogate, i.e. not truly log(pi)

        else:
            logpi = log_softmax_tf(Z, axis=1)

        return P * logpi  # unaggregated, shape: [batch_size, num_actions]

    if dist_id == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution

        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z, ndim=3, axis_size=2, axis=2)
        P.set_shape(Z.get_shape())

        # clip for numerical stability
        P = K.maximum(1e-16, P)
        Z = K.maximum(1e-16, Z)

        # params of behavior policy (will ignore n = alpha + beta)
        p, _ = tf.unstack(P, axis=2)  # shapes: [batch, actions_ndim]

        # params of updateable policy: a = alpha(theta), b = beta(theta)
        p_th, n_th = tf.unstack(Z, axis=2)  # shapes: [batch, actions_ndim]
        a = n_th * p_th
        b = n_th * (1 - p_th)
        ab = K.stack([a, b], -1)

        logpi = -lbeta(ab) + (a - 1) * K.log(p) + (b - 1) * K.log(1 - p)
        return logpi  # shape: [batch_size, actions_ndim]

    raise NotImplementedError("log_pi(dist_id='{}', ...)".format(dist_id))


def proba_ratio(P, Z1, Z2, dist_id):
    """

    This utility function computes the *unaggregated* probability ratio of an
    updateable policy :math:`\\pi_{\\theta_1}(a|s)` and
    :math:`\\pi_{\\theta_2}(a|s)`, evaluated on action(s) sampled from a
    behavior policy :math:`A\\sim b(.|s)`:

    .. math::

        \\rho_\\theta(s, a)\\ =\\
            -b(a|s)\\frac{\\pi_\\theta(a|s)}{\\pi_{\\theta_\\text{old}}(a|s)}

    To turn this into a loss, one needs to aggregate over the batch axis=0 (as
    usual) as well as aggregate/contract over axis=1 (i.e. over the actions
    :math:`a`)

    This is intended to be used as part of a loss/objective function. As such,
    this may actually return a surrogate :math:`\\tilde{H}` instead. Such a
    surrogate is constructed in such a way that it satisfies:

    .. math::

        \\nabla_\\theta \\tilde{\\rho}_\\theta
            \\ =\\ \\nabla_\\theta \\rho_\\theta

    Parameters
    ----------
    P : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        A batch of distribution parameters :term:`P` of the behavior policy.
        For discrete action spaces, this is typically just a one-hot encoded
        version of a batch of taken actions :term:`A`.

    Z1 : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        A batch of distribution parameters. :term:`Z` contains the distribution
        parameters of the updateable policy :math:`\\pi_\\theta(a|s)`.

    Z2 : 2d Tensor, shape: [batch_size, {num_actions,actions_ndim}]

        Same as ``Z1`` except that it comes from a different set of parameters
        :math:`\\theta_2\\neq\\theta_1` (in general).

    dist_id : str

        The policy distribution id, e.g. ``'categorical'`` or ``'beta'`` for
        a softmax policy or a Beta policy, respectively.

    Returns
    -------
    proba_ratio : Tensor, shape: [batch_size, {num_actions,action_ndims}]

        A batch of *unaggregated* probability ratios :math:`\\rho_\\theta(s,
        a)`.

    """
    # # cannot use surrogate, because ratio is not linear in logpi
    # log_pi1 = K.sum(log_pi(P, Z1, dist_id, allow_surrogate=False), axis=1)
    # log_pi2 = K.sum(log_pi(P, Z2, dist_id, allow_surrogate=False), axis=1)
    # return K.exp(log_pi1 - log_pi2)

    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z1, ndim=2)      # params of pi_theta1(a|s)
        check_tensor(Z2, same_as=Z1)  # params of pi_theta2(a|s)
        P.set_shape(Z1.get_shape())   # params of b(a|s)

        # ratio as difference of log probabilities
        logpi1 = log_softmax_tf(Z1, axis=1)
        logpi2 = log_softmax_tf(Z2, axis=1)
        ratio = K.exp(logpi1 - logpi2)

        return P * ratio  # unaggregated, shape: [batch_size, num_actions]

    if dist_id == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution

        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z1, ndim=3, axis_size=2, axis=2)  # pi_theta1(a|s)
        check_tensor(Z2, same_as=Z1)  # params of pi_theta2(a|s)
        P.set_shape(Z1.get_shape())   # params of b(a|s)

        # clip for numerical stability
        P = K.maximum(1e-16, P)
        Z1 = K.maximum(1e-16, Z1)
        Z2 = K.maximum(1e-16, Z2)

        # params of behavior policy (will ignore n = alpha + beta)
        p, _ = tf.unstack(P, axis=2)  # shapes: [batch, actions_ndim]

        # params of policy 1: a1 = alpha(theta_1), b2 = beta(theta_1)
        p1, n1 = tf.unstack(Z1, axis=2)  # shapes: [batch, actions_ndim]
        a1 = n1 * p1
        b1 = n1 * (1 - p1)
        ab1 = K.stack([a1, b1], -1)

        # params of policy 2: a2 = alpha(theta_2), b2 = beta(theta_2)
        p2, n2 = tf.unstack(Z2, axis=2)  # shapes: [batch, actions_ndim]
        a2 = n2 * p2
        b2 = n2 * (1 - p2)
        ab2 = K.stack([a2, b2], -1)

        # ratio as difference of log probabilities
        ratio = K.exp(
            -lbeta(ab1) + lbeta(ab2) + (a1 - a2) * K.log(p)
            + (b1 - b2) * K.log(1 - p))  # noqa: W503

        return ratio  # unaggregated, shape: [batch_size, actions_ndim]

    raise NotImplementedError("log_pi(dist_id='{}', ...)".format(dist_id))


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
