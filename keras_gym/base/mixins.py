import gym
import numpy as np

from .errors import NonDiscreteActionSpace


class RandomStateMixin:
    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        self._random_seed = new_random_seed
        self.random = np.random.RandomState(self._random_seed)

    @random_seed.deleter
    def random_seed(self):
        self._random_seed = None
        self.random = np.random.RandomState(self._random_seed)


class NumActionsMixin:
    @property
    def num_actions(self):
        if not hasattr(self, '_num_actions'):
            if not isinstance(self.env.action_space, gym.spaces.Discrete):
                raise NonDiscreteActionSpace(
                    "num_actions property is inaccesible")
            self._num_actions = self.env.action_space.n
        return self._num_actions
