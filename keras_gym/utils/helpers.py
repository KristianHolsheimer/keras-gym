import os
import time
import logging

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.linalg import pascal
from PIL import Image

from ..base.errors import NumpyArrayCheckError, TensorCheckError


__all__ = (
    'argmax',
    'argmin',
    'check_numpy_array',
    'check_tensor',
    'diff_transform_matrix',
    'enable_logging',
    'feature_vector',
    'generate_gif',
    'get_env_attr',
    'get_transition',
    'has_env_attr',
    'idx',
    'is_policy',
    'is_qfunction',
    'is_vfunction',
    'log_softmax',
    'log_softmax_tf',
    'one_hot',
    'project_onto_actions_np',
    'project_onto_actions_tf',
    'render_episode',
    'set_tf_loglevel',
    'softmax',
)


def enable_logging(silence_tf_logging=True):
    logging.basicConfig(level=logging.INFO)
    if silence_tf_logging:
        set_tf_loglevel(logging.ERROR)


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


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


def render_episode(env, policy, step_delay_ms=0):
    """
    Run a single episode with env.render() calls with each time step.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    policy : callable

        A policy objects that is used to pick actions: ``a = policy(s)``.

    step_delay_ms : non-negative float

        The number of milliseconds to wait between consecutive timesteps. This
        can be used to slow down the rendering.

    """
    s = env.reset()
    env.render()

    for t in range(int(1e9)):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        env.render()
        time.sleep(step_delay_ms / 1e3)

        if done:
            break

        s = s_next

    time.sleep(5 * step_delay_ms / 1e3)
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


def check_tensor(
        tensor,
        ndim=None,
        ndim_min=None,
        dtype=None,
        int_shape=None,
        axis_size=None,
        axis=None):
    """
    This helper function is mostly for internal use. It allows you to check a
    few common properties of a numpy array.

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
    m = s.dot(pascal(num_frames, kind='upper'))[::-1, ::-1]
    return K.constant(m, dtype=dtype)


def has_env_attr(env, attr, max_depth=100):
    """
    Check if a potentially wrapped environment has a given attribute.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return True
        if not hasattr(e, 'env'):
            break
        e = e.env

    return False


def get_env_attr(env, attr, default='__ERROR__', max_depth=100):
    """
    Get the given attribute from a potentially wrapped environment.

    Note that the wrapped envs are traversed from the outside in. Once the
    attribute is found, the search stops. This means that an inner wrapped env
    may carry the same (possibly conflicting) attribute. This situation is
    *not* resolved by this function.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, 'env'):
            break
        e = e.env

    if default == '__ERROR__':
        raise AttributeError("env is missing attribute: {}".format(attr))

    return default


def generate_gif(env, policy, filepath, resize_to=None, duration=50):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment

        The environment to record from.

    policy : keras-gym policy object

        The policy that is used to take actions.

    filepath : str

        Location of the output gif file.

    resize_to : tuple of ints, optional

        The size of the output frames, ``(width, height)``. Notice the
        ordering: first **width**, then **height**. This is the convention PIL
        uses.

    duration : float, optional

        Time between frames in the animated gif, in milliseconds.

    """
    logger = logging.getLogger('generate_gif')

    # collect frames
    frames = []
    s = env.reset()
    for t in range(env.spec.max_episode_steps or 10000):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        # store frame
        frame = info.get('s_orig', [s])[0]
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if len(resize_to) != 2:
                raise TypeError("expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = info.get('s_next_orig', [s])[0]
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    frames[0].save(
        fp=filepath, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=0)

    logger.info("recorded episode to: {}".format(filepath))


def is_vfunction(obj):
    """
    Check whether an object is a :term:`state value function`, or V-function.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a V-function.

    """
    # import at runtime to avoid circular dependence
    from ..function_approximators.base import BaseV
    return isinstance(obj, BaseV)


def is_qfunction(obj, qtype=None):
    """
    Check whether an object is a state-action value function, or Q-function.

    Parameters
    ----------
    obj

        Object to check.

    qtype : 1 or 2, optional

        Check for specific Q-function type, i.e. :term:`type-I <type-I
        state-action value function>` or :term:`type-II <type-II state-action
        value function>`.

    Returns
    -------
    bool

        Whether ``obj`` is a (type-I/II) Q-function.

    """
    # import at runtime to avoid circular dependence
    from ..function_approximators.base import BaseQTypeI, BaseQTypeII

    if qtype is None:
        return isinstance(obj, (BaseQTypeI, BaseQTypeII))
    elif qtype in (1, 1., '1', 'i', 'I'):
        return isinstance(obj, BaseQTypeI)
    elif qtype in (2, 2., '2', 'ii', 'II'):
        return isinstance(obj, BaseQTypeII)
    else:
        raise ValueError("unexpected qtype: {}".format(qtype))


def is_policy(obj, check_updateable=False):
    """
    Check whether an object is a :term:`state value function`, or V-function.

    Parameters
    ----------
    obj

        Object to check.

    check_updateable : bool, optional

        If the obj is a policy, also check whether or not the policy is
        updateable.

    Returns
    -------
    bool

        Whether ``obj`` is a (updateable) policy.

    """
    # import at runtime to avoid circular dependence
    from ..policies.base import BasePolicy
    from ..function_approximators.base import BaseSoftmaxPolicy

    if isinstance(obj, BasePolicy):
        return isinstance(obj, BaseSoftmaxPolicy) if check_updateable else True
    return False
