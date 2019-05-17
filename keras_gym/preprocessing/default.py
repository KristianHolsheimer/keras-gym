import gym
import numpy as np


from ..base.mixins import AddOrigStateToInfoDictMixin


__all__ = (
    'DefaultPreprocessor',
    'feature_vector',
    'one_hot_vector',
)


class DefaultPreprocessor(gym.Wrapper, AddOrigStateToInfoDictMixin):
    """
    This is our default preprocessor. It's an environment wrapper that ensures
    that the state observations can be readily fed into a function
    approximator.

    The original, non-preprocessed state observation is stored in the ``info``
    dict, with the key ``info['s_orig']``. The corresponding value is a list,
    whose individual entries correspond to each consecutive preprocessing step.

    Parameters
    ----------
    env : gym environment

        A gym environment.


    Examples
    --------
    The original state ``s`` and as well as the original next-state ``s_next``
    are stored in the ``info`` dict. For instance, the state observation space
    of the FrozenLake environment is Discrete, which is one-hot encoded by
    :class:`DefaultPreprocessor`. See example below.

    >>> import gym
    >>> from keras_gym.preprocessing import DefaultPreprocessor
    >>> from keras_gym.utils import get_transition
    >>> env = gym.make('FrozenLake-v0')
    >>> env = DefaultPreprocessor(env)
    >>> s = env.reset()
    >>> s_next, r, done, info = env.step(a)
    >>> info
    {'prob': 0.3333333333333333, 's_orig': [0], 's_next_orig': [1]}
    >>> s
    array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> s_next
    array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>>


    """
    def __init__(self, env):
        super().__init__(env)
        s = self.env.observation_space.sample()
        s = feature_vector(s, self.env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
            shape=s.shape)

    def reset(self):
        self._s_orig = self.env.reset()
        s = feature_vector(self._s_orig, self.env.observation_space)
        return s

    def step(self, a):
        self._s_next_orig, r, done, info = self.env.step(a)
        self._add_orig_to_info_dict(info)
        s_next = feature_vector(self._s_next_orig, self.env.observation_space)
        return s_next, r, done, info


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
        x = one_hot_vector(x, space.n)
    elif isinstance(space, (gym.spaces.MultiBinary, gym.spaces.Box)):
        x = x.ravel()
    else:
        raise NotImplementedError(
            "haven't implemented a preprocessor for space type: {}"
            .format(type(space)))

    assert x.ndim == 1, "x must be 1d array, got shape: {}".format(x.shape)
    return x


def one_hot_vector(i, n, dtype='float'):
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
