import logging

import gym
import numpy as np

from .errors import NonDiscreteActionSpace


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)


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


class AddOrigStateToInfoDictMixin:
    def _add_orig_to_info_dict(self, info):
        if not isinstance(info, dict):
            assert info is None, "unexpected type for 'info' dict"
            info = {}

        if 's_orig' in info:
            info['s_orig'].append(self._s_orig)
        else:
            info['s_orig'] = [self._s_orig]

        if 's_next_orig' in info:
            info['s_next_orig'].append(self._s_next_orig)
        else:
            info['s_next_orig'] = [self._s_next_orig]

        self._s_orig = self._s_next_orig
