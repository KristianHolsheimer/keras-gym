import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.linalg

from ..base.errors import TensorCheckError, SpaceError


__all__ = (
    'box_to_reals_tf',
    'box_to_unit_interval_tf',
    'check_tensor',
    'diff_transform_matrix',
    'log_softmax_tf',
    'project_onto_actions_tf',
    'reals_to_box_tf',
    'unit_interval_to_box_tf',
)


def box_to_unit_interval_tf(tensor, space):
    """

    Rescale Tensor values from Box space to the unit interval. This is
    essentially just min-max scaling:

    .. math::

        x\\ \\mapsto\\ \\frac{x-x_\\text{low}}{x_\\text{high}-x_\\text{low}}

    Parameters
    ----------
    tensor : nd Tensor

        A tensor containing a single instance or a batch of elements of a Box
        space.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd Tensor, same shape as input

        A Tensor with the transformed values. The output values lie on the unit
        interval :math:`[0,1]`.

    """
    tensor, lo, hi = _get_box_bounds(tensor, space)
    return (tensor - lo) / (hi - lo)


def box_to_reals_tf(tensor, space, epsilon=1e-15):
    """

    Transform Tensor values from a Box space to the reals. This is done by
    first mapping the Box values to the unit interval :math:`x\\in[0, 1]` and
    then feeding it to the :func:`clipped_logit_tf` function.

    Parameters
    ----------
    tensor : nd Tensor

        A tensor containing a single instance or a batch of elements of a Box
        space.


    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    epsilon : float, optional

        The cut-off value used by :func:`clipped_logit_tf`.

    Returns
    -------
    out : nd Tensor, same shape as input

        A Tensor with the transformed values. The output values are
        real-valued.

    """
    return clipped_logit_tf(box_to_unit_interval_tf(tensor, space), epsilon)


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
    ndim : int or list of ints

        Check ``K.ndim(tensor)``.

    ndim_min : int

        Check if ``K.ndim(tensor)`` is at least ``ndim_min``.

    dtype : Tensor dtype or list of Tensor dtypes

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
    ndims = [ndim] if not isinstance(ndim, (list, tuple, set)) else ndim
    if check and K.ndim(tensor) not in ndims:
        raise TensorCheckError(
            "expected input with ndim(s) equal to {}, got ndim: {}"
            .format(ndim, K.ndim(tensor)))

    check = ndim_min is not None
    if check and K.ndim(tensor) < ndim_min:
        raise TensorCheckError(
            "expected input with ndim at least {}, got ndim: {}"
            .format(ndim_min, K.ndim(tensor)))

    check = dtype is not None
    dtypes = [dtype] if not isinstance(dtype, (list, tuple, set)) else dtype
    if check and K.dtype(tensor) not in dtypes:
        raise TensorCheckError(
            "expected input with dtype(s) {}, got dtype: {}"
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


def clipped_logit_tf(x, epsilon=1e-15):
    """

    A safe implementation of the logit function
    :math:`x\\mapsto\\log(x/(1-x))`. It clips the arguments of the log function
    from below so as to avoid evaluating it at 0:

    .. math::

        \\text{logit}_\\epsilon(x)\\ =\\
            \\log(\\max(\\epsilon, x)) - \\log(\\max(\\epsilon, 1 - x))

    Parameters
    ----------
    x : nd Tensor

        Input tensor whose entries lie on the unit interval, :math:`x_i\\in
        [0,1]`.

    epsilon : float, optional

        The small number with which to clip the arguments of the logarithm from
        below.

    Returns
    -------
    z : nd Tensor, dtype: float, shape: same as input

        The output logits whose entries lie on the real line,
        :math:`z_i\\in\\mathbb{R}`.

    """
    return K.log(K.maximum(epsilon, x)) - K.log(K.maximum(epsilon, 1 - x))


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


def reals_to_box_tf(tensor, space):
    """

    Transform Tensor values from the reals to a Box space. This is done by
    first applying the logistic sigmoid to map the reals onto the unit interval
    and then applying :func:`unit_interval_to_box_tf` to rescale to the Box
    space.

    Parameters
    ----------
    tensor : nd Tensor

        A tensor containing a single instance or a batch of elements of a Box
        space, encoded as logits.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd Tensor, same shape as input

        A Tensor with the transformed values. The output values are contained
        in the provided Box space.

    """
    return unit_interval_to_box_tf(K.sigmoid(tensor), space)


def unit_interval_to_box_tf(tensor, space):
    """

    Rescale Tensor values from the unit interval to a Box space. This is
    essentially `inverted` min-max scaling:

    .. math::

        x\\ \\mapsto\\ x_\\text{low} + (x_\\text{high} - x_\\text{low})\\,x

    Parameters
    ----------
    tensor : nd Tensor

        A numpy array containing a single instance or a batch of elements of
        a Box space, scaled to the unit interval.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd Tensor, same shape as input

        A Tensor with the transformed values. The output values are contained
        in the provided Box space.

    """
    arr, lo, hi = _get_box_bounds(tensor, space)
    return lo + (hi - lo) * arr


def _get_box_bounds(tensor, space):
    check_tensor(tensor, dtype=('float32', 'float64'))
    if not isinstance(space, gym.spaces.Box):
        raise SpaceError("space must be a Box")

    # prepare box bounds
    lo = K.constant(space.low, dtype=tensor.dtype)
    hi = K.constant(space.high, dtype=tensor.dtype)

    if K.ndim(tensor) == K.ndim(lo) + 1:
        shape = K.int_shape(tensor)[1:]
        lo = K.expand_dims(lo, axis=0)
        hi = K.expand_dims(hi, axis=0)
    else:
        shape = K.int_shape(tensor)

    if shape != K.int_shape(lo) or shape != K.int_shape(hi):
        SpaceError("tensor shape is incompatible with the Box space")

    return tensor, lo, hi
