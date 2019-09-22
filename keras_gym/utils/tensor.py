import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.linalg

from ..base.errors import TensorCheckError
from .misc import check_dist_id


__all__ = (
    'check_tensor',
    'cross_entropy',
    'diff_transform_matrix',
    'entropy',
    'log_softmax_tf',
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


def cross_entropy(
        Z1, Z2, dist_id,
        allow_surrogate=True,
        Z1_is_logit=True,
        Z2_is_logit=True):

    """

    This utility function computes the cross-entropy of an updateable policy
    :math:`\\pi_{\\theta_2}(a|s)` relative to a another policy
    :math:`\\pi_{\\theta_1}(a|s)`:

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

        The policy distribution id, e.g. ``'categorical'`` or ``'normal'`` for
        a softmax policy or a Gaussian policy, respectively.

    allow_surrogate : bool, optional

        Whether to allow the function to return a surrogate function instead of
        the true function. The surrogate is generally more numerically stable,
        but if your loss/objective is of the form :math:`J(\\theta) =
        H[b,\\pi_\\theta]\\,f(\\theta)` it breaks the surrogate condition, i.e.
        :math:`\\nabla_\\theta \\tilde{J}(\\theta) \\neq \\nabla_\\theta
        J(\\theta)`.

    Z1_is_logit : bool, optional

        A flag to indicate whether the input ``Z1`` holds logits or whether it
        already has had the softmax applied to it. This argument only applies
        if the ``dist_id='categorical'``. It's ignored otherwise.

    Z2_is_logit : bool, optional

        A flag to indicate whether the input ``Z2`` holds logits or whether it
        already has had the softmax applied to it. This argument only applies
        if the ``dist_id='categorical'``. It's ignored otherwise.

    Returns
    -------
    cross_entropy : Tensor, shape: [batch_size]

        A batch of cross-entropy values :math:`H[b,\\pi_\\theta](s, a)`.

    """
    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z1, ndim=2)
        check_tensor(Z2, same_as=Z1)

        if allow_surrogate:
            assert Z2_is_logit
            # Construct surrogate:
            # Let pi(a|s) = softmax(z(s,a)), then the surrogate for log(pi) is
            # logpi_surrogate = z(s,a) - sum_a' stop_gradient(pi(a'|s)) z(s,a')
            pi2 = K.stop_gradient(K.softmax(Z2, axis=1))
            Z2_mean = K.expand_dims(tf.einsum('ij,ij->i', pi2, Z2), axis=1)
            logpi2 = Z2 - Z2_mean  # surrogate, i.e. not truly log(pi)
        else:
            logpi2 = log_softmax_tf(Z2, axis=1) if Z2_is_logit else K.log(Z2)

        pi1 = K.softmax(Z1, axis=1) if Z1_is_logit else Z1
        return K.sum(-pi1 * logpi2, axis=1)  # shape: [batch_size]

    if dist_id == 'normal':
        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z1, ndim=3, axis_size=2, axis=2)
        check_tensor(Z2, same_as=Z1)

        # extract params
        mu1, logvar1 = tf.unstack(Z1, axis=2)  # shapes: [batch, actions_ndim]
        mu2, logvar2 = tf.unstack(Z2, axis=2)  # shapes: [batch, actions_ndim]

        cross_entropy = (
            K.exp(logvar1 - logvar2)
            + K.square(mu1 - mu2) / K.exp(logvar2)
            + logvar2 + K.log(2 * np.pi)) / 2

        return K.mean(cross_entropy, axis=1)  # shape: [batch_size]

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


def entropy(Z, dist_id, Z_is_logit=True):
    """

    This utility function computes the entropy of an updateable policy
    :math:`\\pi_\\theta(a|s)`.

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

        The policy distribution id, e.g. ``'categorical'`` or ``'normal'`` for
        a softmax policy or a Gaussian policy, respectively.

    Z_is_logit : bool, optional

        A flag to indicate whether the input ``Z`` holds logits or whether it
        already has had the softmax applied to it. This argument only applies
        if the ``dist_id='categorical'``. It's ignored otherwise.

    Returns
    -------
    entropy : Tensor, shape: [batch_size]

        A batch of entropy values :math:`H[\\pi_\\theta](s, a)`.

    """
    dist_id = check_dist_id(dist_id)

    if dist_id == 'categorical':
        # expected input shapes: [batch_size, num_actions]
        check_tensor(Z, ndim=2)

        # action propensities
        if Z_is_logit:
            pi = K.softmax(Z, axis=1)
            logpi = log_softmax_tf(Z, axis=1)
        else:
            pi = Z
            logpi = K.log(pi)

        return K.sum(-pi * logpi, axis=1)  # shape: [batch_size]

    if dist_id == 'normal':
        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z, ndim=3, axis_size=2, axis=2)

        # dist params of policy
        mu, logvar = tf.unstack(Z, axis=2)  # shapes: [batch, actions_ndim]

        # entropy.shape: [batch_size, actions_ndim]
        entropy = (1. + K.log(2 * np.pi) + logvar) / 2.

        return K.mean(entropy, axis=1)  # shape: [batch_size]

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


def proba_ratio(P, Z1, Z2, dist_id):
    """

    This utility function computes the probability ratio of an updateable
    policy :math:`\\pi_{\\theta_1}(a|s)` and :math:`\\pi_{\\theta_2}(a|s)`,
    evaluated on action(s) sampled from a behavior policy :math:`A\\sim
    b(.|s)`:

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

        The policy distribution id, e.g. ``'categorical'`` or ``'normal'`` for
        a softmax policy or a Gaussian policy, respectively.

    Returns
    -------
    proba_ratio : Tensor, shape: [batch_size]

        A batch of probability ratios :math:`\\rho_\\theta(s, a)`.

    """
    dist_id = check_dist_id(dist_id)

    extra_kwargs = {}
    if dist_id == 'categorical':
        check_tensor(Z1, ndim=2)      # params of pi_theta1(a|s)
        check_tensor(Z2, same_as=Z1)  # params of pi_theta2(a|s)
        P.set_shape(Z1.get_shape())   # params of b(a|s)
        extra_kwargs.update({'Z1_is_logit': False, 'allow_surrogate': False})
    elif dist_id == 'normal':
        # expected input shapes: [batch_size, actions_ndim, 2]
        check_tensor(Z1, ndim=3, axis_size=2, axis=2)  # pi_theta1(a|s)
        check_tensor(Z2, same_as=Z1)  # params of pi_theta2(a|s)
        P.set_shape(Z1.get_shape())   # params of b(a|s)
    else:
        raise NotImplementedError(
            "proba_ratio(dist_id='{}', ...)".format(dist_id))

    # ratio as difference of log probabilities
    logpi1 = -cross_entropy(P, Z1, dist_id, **extra_kwargs)
    logpi2 = -cross_entropy(P, Z2, dist_id, **extra_kwargs)
    ratio = K.exp(logpi1 - logpi2)

    return ratio  # shape: [batch_size]


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
