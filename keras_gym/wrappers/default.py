import gym
import numpy as np


from ..base.mixins import AddOrigStateToInfoDictMixin
from ..utils import one_hot, feature_vector


__all__ = (
    'DefaultPreprocessor',
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
    >>> import keras_gym as km
    >>> env = gym.make('FrozenLake-v0')
    >>> env = km.wrappers.DefaultPreprocessor(env)
    >>> s = env.reset()
    >>> a = env.action_space.sample()
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
