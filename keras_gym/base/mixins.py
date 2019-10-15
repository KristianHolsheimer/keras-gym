import logging

import gym
import numpy as np

from ..utils import one_hot
from .errors import ActionSpaceError


class LoggerMixin:
    # try:
    #     """
    #     This workaround was taken from here:

    #         https://github.com/dhalperi/pybatfish/blob/f8ddd3938148f9a5d9c14c371a099802c564fac3/pybatfish/client/capirca.py#L33-L50

    #     As of version 1.14, Tensorflow uses Google's abseil-py library, which
    #     uses a Google-specific wrapper for logging. That wrapper will write a
    #     warning to sys.stderr if the Google command-line flags library has
    #     not been initialized.

    #         https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825

    #     This is not right behavior for Python code that is invoked outside of
    #     a Google-authored main program. Use knowledge of abseil-py to disable
    #     that warning; ignore and continue if something goes wrong.

    #     """  # noqa: E501
    #     import absl.logging

    #     # https://github.com/abseil/abseil-py/issues/99
    #     logging.root.removeHandler(absl.logging._absl_handler)

    #     # https://github.com/abseil/abseil-py/issues/102
    #     absl.logging._warn_preinit_stderr = False

    # except Exception:
    #     pass

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


class ActionSpaceMixin:
    @property
    def action_space_is_box(self):
        return isinstance(self.env.action_space, gym.spaces.Box)

    @property
    def action_space_is_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def actions_ndim(self):
        if not hasattr(self, '_actions_ndim'):
            if not self.action_space_is_box:
                raise ActionSpaceError(
                    "actions_ndim attribute is inaccesible; does the env "
                    "have a Box action space?")
            self._actions_ndim = len(self.env.action_space.shape) or 1
        return self._actions_ndim

    @property
    def num_actions(self):
        if not hasattr(self, '_num_actions'):
            if not self.action_space_is_discrete:
                raise ActionSpaceError(
                    "num_actions attribute is inaccesible; does the env have "
                    "a Discrete action space?")
            self._num_actions = self.env.action_space.n
        return self._num_actions

    def _one_hot_encode_discrete(self, a):
        # no-op if 'a' already looks one-hot encoded
        if not (np.ndim(a) == 1 and a.size == self.num_actions):
            assert np.ndim(a) == 0 and isinstance(a, (int, np.integer))
            a = one_hot(a, self.num_actions)
        return a


class AddOrigToInfoDictMixin:
    def _add_s_orig_to_info_dict(self, info):
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

    def _add_a_orig_to_info_dict(self, info):
        if not isinstance(info, dict):
            assert info is None, "unexpected type for 'info' dict"
            info = {}

        if 'a_orig' in info:
            info['a_orig'].append(self._a_orig)
        else:
            info['a_orig'] = [self._a_orig]
