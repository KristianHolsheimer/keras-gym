import gym
import numpy as np

from scipy.special import expit as sigmoid

from ..base.errors import NumpyArrayCheckError, SpaceError


__all__ = (
    'argmax',
    'argmin',
    'batch_to_single_instance',
    'box_to_reals_np',
    'box_to_unit_interval_np',
    'check_numpy_array',
    'clipped_logit_np',
    'feature_vector',
    'idx',
    'log_softmax',
    'one_hot',
    'project_onto_actions_np',
    'reals_to_box_np',
    'softmax',
    'unit_interval_to_box_np',
)


def argmax(arr, axis=-1, random_state=None):
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


def batch_to_single_instance(X):
    """
    Take the first instance of an array that contains a batch of items.

    Parameters
    ----------
    X : nd array, shape: [batch_size, \\*instance_shape]

        A numpy array whose first axis is the batch axis.

    Returns
    -------
    x : (n-1)d array, shape: [\\*instance_shape]

        This is essentially just ``X[0]`` with some post processing.

    """
    x = X[0]
    assert False, x.ndim
    if x.ndim == 0 and x.dtype == 'int':
        x = int(x.item())
    if x.ndim == 0 and x.dtype == 'float':
        x = float(x.item())
    return x


def box_to_unit_interval_np(arr, space):
    """

    Rescale array values from Box space to the unit interval. This is
    essentially just min-max scaling:

    .. math::

        x\\ \\mapsto\\ \\frac{x-x_\\text{low}}{x_\\text{high}-x_\\text{low}}

    Parameters
    ----------
    arr : nd array

        A numpy array containing a single instance or a batch of elements of
        a Box space.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd array, same shape as input

        A numpy array with the transformed values. The output values lie on the
        unit interval :math:`[0, 1]`.

    """
    arr, lo, hi = _get_box_bounds(arr, space)

    # box to unit interval
    p = (arr - lo) / (hi - lo)

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("array values are not contained in the provided Box")

    return p


def box_to_reals_np(arr, space, epsilon=1e-15):
    """

    Transform array values from a Box space to the reals. This is done by
    first mapping the Box values to the unit interval :math:`x\\in[0, 1]` and
    then feeding it to the :func:`clipped_logit_np` function.

    Parameters
    ----------
    arr : nd array

        A numpy array containing a single instance or a batch of elements of
        a Box space.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    epsilon : float, optional

        The cut-off value used by :func:`clipped_logit_np`.

    Returns
    -------
    out : nd array, same shape as input

        A numpy array with the transformed values. The output values are
        real-valued.

    """
    return clipped_logit_np(box_to_unit_interval_np(arr, space), epsilon)


def check_numpy_array(arr, ndim=None, ndim_min=None, dtype=None, shape=None, axis_size=None, axis=None):  # noqa: E501
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


def clipped_logit_np(x, epsilon=1e-15):
    """

    A safe implementation of the logit function
    :math:`x\\mapsto\\log(x/(1-x))`. It clips the arguments of the log function
    from below so as to avoid evaluating it at 0:

    .. math::

        \\text{logit}_\\epsilon(x)\\ =\\
            \\log(\\max(\\epsilon, x)) - \\log(\\max(\\epsilon, 1 - x))

    Parameters
    ----------
    x : nd array

        Input numpy array whose entries lie on the unit interval,
        :math:`x_i\\in [0, 1]`.

    epsilon : float, optional

        The small number with which to clip the arguments of the logarithm from
        below.

    Returns
    -------
    z : nd array, dtype: float, shape: same as input

        The output logits whose entries lie on the real line,
        :math:`z_i\\in\\mathbb{R}`.

    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("values do not lie on the unit interval")
    return np.log(np.maximum(epsilon, x)) - np.log(np.maximum(epsilon, 1 - x))


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
    i : int or 1d array of ints

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
    if isinstance(i, (int, np.integer)):
        if not 0 <= i < n:
            raise ValueError("i must be non-negative and smaller than n")
        x = np.zeros(int(n), dtype=dtype)
        x[int(i)] = 1.0
        return x

    if isinstance(i, np.ndarray) and i.ndim == 1 and i.dtype == np.integer:
        if np.any(i >= n) or np.any(i < 0):
            raise ValueError("i must be non-negative and smaller than n")
        x = np.zeros((len(i), int(n)), dtype=dtype)
        x[idx(i), i] = 1.0
        return x

    raise ValueError("i must be an int or 1d array of ints")


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


def reals_to_box_np(arr, space):
    """

    Transform array values from the reals to a Box space. This is done by first
    applying the logistic sigmoid to map the reals onto the unit interval and
    then applying :func:`unit_interval_to_box_np` to rescale to the Box
    space.

    Parameters
    ----------
    arr : nd array

        A numpy array containing a single instance or a batch of elements of
        a Box space, encoded as logits.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd array, same shape as input

        A numpy array with the transformed values. The output values are
        contained in the provided Box space.

    """
    return unit_interval_to_box_np(sigmoid(arr), space)


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


def unit_interval_to_box_np(arr, space):
    """

    Rescale array values from the unit interval to a Box space. This is
    essentially `inverted` min-max scaling:

    .. math::

        x\\ \\mapsto\\ x_\\text{low} + (x_\\text{high} - x_\\text{low})\\,x

    Parameters
    ----------
    arr : nd array

        A numpy array containing a single instance or a batch of elements of
        a Box space, scaled to the unit interval.

    space : gym.spaces.Box

        The Box space. This is needed to determine the shape and size of the
        space.

    Returns
    -------
    out : nd array, same shape as input

        A numpy array with the transformed values. The output values are
        contained in the provided Box space.

    """
    arr, lo, hi = _get_box_bounds(arr, space)
    return lo + (hi - lo) * arr


def _get_box_bounds(arr, space):
    check_numpy_array(arr, dtype=('float', np.float32, np.float64))
    if not isinstance(space, gym.spaces.Box):
        raise SpaceError("space must be a Box")

    # prepare box bounds
    lo, hi = space.low, space.high
    check_numpy_array(lo, dtype=('float', np.float32, np.float64))
    check_numpy_array(hi, dtype=('float', np.float32, np.float64))
    if np.ndim(arr) == np.ndim(lo) + 1:
        shape = arr.shape[1:]
        lo = np.expand_dims(lo, axis=0)
        hi = np.expand_dims(hi, axis=0)
    else:
        shape = arr.shape

    if shape != lo.shape or shape != hi.shape:
        SpaceError("array shape is incompatible with the Box space")

    return arr, lo, hi
