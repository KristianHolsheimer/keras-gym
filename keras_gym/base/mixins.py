import logging

import gym
import numpy as np

from ..utils import one_hot, check_numpy_array
from .errors import ActionSpaceError


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

    def check_a_or_params(self, a_or_params):
        """
        Check if input ``pi`` is either a valid vector of action propensities.
        If ``pi`` is an integer, this will return a one-hot encoded version.

        Parameters
        ----------
        a_or_params : action or distribution parameters

            Either a single action taken under the behavior policy or a single
            set of distribution parameters describing the behavior policy
            :math:`b(a|s)`. See also the glossary entry for :term:`P`.

            For instance, let's say our action space is :class:`Discrete(4)`,
            then passing ``a_or_params = 2`` is equivalent to passing
            ``a_or_params = [0, 0, 1, 0]``. Both would indicate that the action
            :math:`a=2` was drawn from the behavior policy.

        Returns
        -------
        params : nd array

            A single set of distribution parameters describing the behavior
            policy :math:`b(a|s)`. This is either same or derived from the
            input.

        """
        if self.action_space_is_discrete:

            if self.env.action_space.contains(a_or_params):
                assert self.env.action_space.contains(a_or_params)
                # assume Categorical distribution
                params = one_hot(a_or_params, self.num_actions)
            else:
                check_numpy_array(
                    a_or_params, ndim=1, axis_size=self.num_actions, axis=0)
                params = a_or_params

        elif self.action_space_is_box:

            if self.env.action_space.contains(a_or_params):
                # assume Beta(alpha, beta) distribution
                p = a_or_params  # p == alpha / (alpha + beta)
                n = np.infty     # n = alpha + beta
                params = p, n
            else:
                check_numpy_array(
                    a_or_params, ndim=1, axis_size=self.actions_ndim, axis=0)
                alpha, beta = a_or_params
                n = alpha + beta
                p = alpha / n
                params = p, n
        else:
            raise ActionSpaceError(
                "check_a_or_params() hasn't yet been implemented for action "
                "spaces of type: {}"
                .format(self.env.action_space.__class__.__name__))

        return params


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
