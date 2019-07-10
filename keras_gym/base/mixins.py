import logging

import gym
import numpy as np

from ..utils import one_hot, check_numpy_array
from .errors import NonDiscreteActionSpace


class LoggerMixin:
    try:
        """
        This workaround was taken from here:

            https://github.com/dhalperi/pybatfish/blob/f8ddd3938148f9a5d9c14c371a099802c564fac3/pybatfish/client/capirca.py#L33-L50

        As of version 1.14, Tensorflow uses Google's abseil-py library, which
        uses a Google-specific wrapper for logging. That wrapper will write a
        warning to sys.stderr if the Google command-line flags library has
        not been initialized.

            https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825

        This is not right behavior for Python code that is invoked outside of
        a Google-authored main program. Use knowledge of abseil-py to disable
        that warning; ignore and continue if something goes wrong.

        """  # noqa: E501
        import absl.logging

        # https://github.com/abseil/abseil-py/issues/99
        logging.root.removeHandler(absl.logging._absl_handler)

        # https://github.com/abseil/abseil-py/issues/102
        absl.logging._warn_preinit_stderr = False

    except Exception:
        pass

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

    def check_pi(self, pi):
        """
        Check if input ``pi`` is either a valid vector of action propensities.
        If ``pi`` is an integer, this will return a one-hot encoded version.

        Parameters
        ----------
        pi : int or 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. This may
            be just an indicator if the action propensities are inferred
            through sampling. For instance, let's say our action space is
            :class:`Discrete(4)`, then passing ``pi = 2`` is equivalent to
            passing ``pi = [0, 0, 1, 0]``. Both would indicate that the action
            :math:`a=2` was drawn from the behavior policy.

        Returns
        -------
        pi : 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. If the
            input ``pi`` is an integer, the output will be a one-hot encoded
            vector.

        """
        if isinstance(pi, (int, np.integer)):
            assert self.env.action_space.contains(pi)
            pi = one_hot(pi, self.num_actions)
        check_numpy_array(pi, ndim=1, axis_size=self.num_actions, axis=0)
        return pi


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
