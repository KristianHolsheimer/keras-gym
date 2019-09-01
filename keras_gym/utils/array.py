import gym
import numpy as np

from ..base.errors import NumpyArrayCheckError


__all__ = (
    'argmax',
    'argmin',
    'check_numpy_array',
    'feature_vector',
    'idx',
    'log_softmax',
    'one_hot',
    'project_onto_actions_np',
    'softmax',
)


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
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 1:
        candidates = np.arange(arr.size)             # all
        candidates = candidates[arr == np.max(arr)]  # max
        if not isinstance(random_state, np.random.RandomState):
            # treat input random_state as seed
            random_state = np.random.RandomState(random_state)
        return random_state.choice(candidates)
    else:
        return np.argmax(arr, axis=axis)


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
    return argmax(-arr, axis=axis, random_state=random_state)


def check_numpy_array(arr, ndim=None, ndim_min=None, dtype=None, shape=None, axis_size=None, axis=None):  # nowqa: E501
    """

    This helper function is mostly for internal use. It is used to check a few
    common properties of a numpy array.

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
    ndims = [ndim] if not isinstance(ndim, (list, tuple, set)) else ndim
    if check and arr.ndim not in ndims:
        raise NumpyArrayCheckError(
            "expected input with ndim(s) {}, got ndim: {}"
            .format(ndim, arr.ndim))

    check = ndim_min is not None
    if check and arr.ndim < ndim_min:
        raise NumpyArrayCheckError(
            "expected input with ndim at least {}, got ndim: {}"
            .format(ndim_min, arr.ndim))

    check = dtype is not None
    dtypes = [dtype] if not isinstance(dtype, (list, tuple, set)) else dtype
    if check and arr.dtype not in dtypes:
        raise NumpyArrayCheckError(
            "expected input with dtype(s) {}, got dtype: {}"
            .format(dtype, arr.dtype))

    check = shape is not None
    if check and arr.shape != shape:
        raise NumpyArrayCheckError(
            "expected input with shape {}, got shape: {}"
            .format(shape, arr.shape))

    check = axis_size is not None and axis is not None
    sizes = [axis_size] if not isinstance(axis_size, (list, tuple, set)) else axis_size  # noqa: E501
    if check and arr.shape[axis] not in sizes:
        raise NumpyArrayCheckError(
            "expected input with size(s) {} along axis {}, got shape: {}"
            .format(axis_size, axis, arr.shape))


def feature_vector(x, space):
    """
    Create a feature vector out of a state observation :math:`s` or an action
    :math:`a`. This is used in the :class:`DefaultPreprocessor`.

    Parameters
    ----------
    x : state or action

        A state observation :math:`s` or an action :math:`a`.

    space : gym space

        A gym space, e.g. :class:`gym.spaces.Box`,
        :class:`gym.spaces.Discrete`, etc.

    """
    if space is None:
        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise TypeError(
                "if space is None, x must be a 1d numpy array already")
    elif isinstance(space, gym.spaces.Tuple):
        x = np.concatenate([
            feature_vector(x_, space_)  # recursive
            for x_, space_ in zip(x, space.spaces)], axis=0)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        x = np.concatenate([
            feature_vector(x_, gym.spaces.Discrete(n))  # recursive
            for x_, n in zip(x.ravel(), space.nvec.ravel()) if n], axis=0)
    elif isinstance(space, gym.spaces.Discrete):
        x = one_hot(x, space.n)
    elif isinstance(space, (gym.spaces.MultiBinary, gym.spaces.Box)):
        x = x.ravel()
    else:
        raise NotImplementedError(
            "haven't implemented a preprocessor for space type: {}"
            .format(type(space)))

    assert x.ndim == 1, "x must be 1d array, got shape: {}".format(x.shape)
    return x


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


def log_softmax(arr, axis=-1):
    """
    Compute the log-softmax.

    **Note:** This is the *numpy* implementation.

    Parameters
    ----------
    arr : numpy array

        The input array.

    axis : int, optional

        The axis along which to normalize, default is 0.

    Returns
    -------
    out : array of same shape

        The entries may be interpreted as log-probabilities.

    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    Z = arr - arr.mean(axis=axis, keepdims=True)  # center before clipping
    Z = np.clip(Z, -30, 30)                       # avoid over/underflow
    log_P = Z - np.log(np.sum(np.exp(Z), axis=axis, keepdims=True))
    return log_P


def one_hot(i, n, dtype='float'):
    """
    Create a dense one-hot encoded vector.

    Parameters
    ----------
    i : int

        The index of the non-zero entry.

    n : int

        The dimensionality of the dense vector. Note that `n` must be greater
        than `i`.

    dtype : str or datatype

        The output data type, default is `'float'`.

    Returns
    -------
    x : 1d array of length n

        The dense one-hot encoded vector.

    """
    if not 0 <= i < n:
        raise ValueError("i must be a non-negative and smaller than n")
    x = np.zeros(int(n), dtype=dtype)
    x[int(i)] = 1.0
    return x


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


def softmax(arr, axis=-1):
    """
    Compute the softmax (normalized point-wise exponential).

    **Note:** This is the *numpy* implementation.

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
    Z = arr - np.mean(arr, axis=axis, keepdims=True)  # center before clipping
    Z = np.clip(Z, -30, 30)                           # avoid over/underflow
    exp_Z = np.exp(Z)
    p = exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)
    return p
