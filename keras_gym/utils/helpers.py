import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ..base.errors import NumpyArrayCheckError


__all__ = (
    'argmax',
    'argmin',
    'check_numpy_array',
    'get_transition',
    'idx',
    'project_onto_actions_np',
    'project_onto_actions_tf',
    'softmax',
)


def get_transition(env):
    """
    Generate a transition from the environment.

    This basically does a single step on the environment
    and then closes it.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    Returns
    -------
    s, a, r, s_next, a_next, done, info : tuple

        A single transition. Note that the order and the number of items
        returned is different from what ``env.reset()`` return.

    """
    try:
        s = env.reset()
        a = env.action_space.sample()
        a_next = env.action_space.sample()
        s_next, r, done, info = env.step(a)
        return s, a, r, s_next, a_next, done, info
    finally:
        env.close()


def idx(arr, axis=0):
    """
    Given a numpy array, return its corresponding integer index array.

    Parameters
    ----------
    arr : array
        Input array.

    axis : int, optional
        The axis along which we'd like to get an index.

    Returns
    -------
    index : 1d array, shape: arr.shape[axis]
        An index array `[0, 1, 2, ...]`.

    """
    check_numpy_array(arr, ndim_min=1)
    return np.arange(arr.shape[axis])


def check_numpy_array(
        arr,
        ndim=None,
        ndim_min=None,
        dtype=None,
        shape=None,
        axis_size=None,
        axis=None):
    """
    This helper function is mostly for internal use. It allows you to check a
    few common properties of a numpy array.

    Raises
    ------
    NumpyArrayCheckError

        If one of the checks fails, it raises a :class:`NumpyArrayCheckError`.

    """

    if not isinstance(arr, np.ndarray):
        raise NumpyArrayCheckError(
            "expected input to be a numpy array, got type: {}"
            .format(type(arr)))

    check = ndim is not None
    if check and arr.ndim != ndim:
        raise NumpyArrayCheckError(
            "expected input with ndim equal to {}, got ndim: {}"
            .format(ndim, arr.ndim))

    check = ndim_min is not None
    if check and arr.ndim < ndim_min:
        raise NumpyArrayCheckError(
            "expected input with ndim at least {}, got ndim: {}"
            .format(ndim_min, arr.ndim))

    check = dtype is not None
    if check and arr.dtype != dtype:
        raise NumpyArrayCheckError(
            "expected input with dtype {}, got dtype: {}"
            .format(dtype, arr.dtype))

    check = shape is not None
    if check and arr.shape != shape:
        raise NumpyArrayCheckError(
            "expected input with shape {}, got shape: {}"
            .format(shape, arr.shape))

    check = axis_size is not None and axis is not None
    if check and arr.shape[axis] != axis_size:
        raise NumpyArrayCheckError(
            "expected input with size {} along axis {}, got shape: {}"
            .format(axis_size, axis, arr.shape))


def project_onto_actions_np(Y, A):
    """
    Project tensor onto specific actions taken: **numpy** implementation.

    **Note**: This only applies to discrete action spaces.

    Parameters
    ----------
    Y : 2d array, shape: [batch_size, num_actions]

        The tensor to project down.

    A : 1d array, shape: [batch_size]

        The batch of actions used to project.

    Returns
    -------
    Y_projected : 1d array, shape: [batch_size]

        The tensor projected onto the actions taken.

    """
    check_numpy_array(Y, ndim=2)
    check_numpy_array(A, ndim=1, dtype='int')
    check_numpy_array(Y, axis_size=A.shape[0], axis=0)  # same batch size
    return Y[idx(Y), A]


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


def argmin(arr, axis=None, random_state=None):
    """
    This is a little hack to ensure that argmin breaks ties randomly, which is
    something that :func:`numpy.argmin` doesn't do.

    *Note: random tie breaking is only done for 1d arrays; for multidimensional
    inputs, we fall back to the numpy version.*

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.

    random_state : int or RandomState
        This can either be a random seed (`int`) or an instance of
        :class:`numpy.random.RandomState`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 1:
        candidates = np.arange(arr.size)             # all
        candidates = candidates[arr == np.min(arr)]  # min
        if not isinstance(random_state, np.random.RandomState):
            # treat input random_state as seed
            random_state = np.random.RandomState(random_state)
        return random_state.choice(candidates)
    else:
        return np.argmin(arr, axis=axis)


def argmax(arr, axis=None, random_state=None):
    """
    This is a little hack to ensure that argmax breaks ties randomly, which is
    something that :func:`numpy.argmax` doesn't do.

    *Note: random tie breaking is only done for 1d arrays; for multidimensional
    inputs, we fall back to the numpy version.*

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.

    random_state : int or RandomState
        This can either be a random seed (`int`) or an instance of
        :class:`numpy.random.RandomState`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    return argmin(-arr, axis=axis, random_state=random_state)


def softmax(arr, axis=0):
    """
    Compute the softmax (normalized point-wise exponential).

    Parameters
    ----------
    arr : numpy array
        The input array.

    axis : int, optional
        The axis along which to normalize, default is 0.

    Returns
    -------
    out : array of same shape
        The entries of the output array are non-negative and normalized, which
        make them good candidates for modeling probabilities.

    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr -= arr.mean(axis=axis, keepdims=True)  # center before clipping
    arr = np.clip(arr, -30, 30)                # avoid overflow before exp
    arr = np.exp(arr)
    arr /= arr.sum(axis=axis, keepdims=True)
    return arr
