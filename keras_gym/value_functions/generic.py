import numpy as np

from ..errors import NonDiscreteActionSpaceError
from ..utils import feature_vector

from .base import BaseValueFunction


class GenericV(BaseValueFunction):
    """
    TODO docs

    """
    output_dims = 1

    def __call__(self, s):
        """
        TODO docs

        """
        X = self.X(s)
        V = self.batch_eval(X)
        return V[0]

    def X(self, s):
        """
        TODO docs

        """
        x = feature_vector(s, self.env.observation_space)
        X = np.expand_dims(x, axis=0)
        return X


class GenericQ(BaseValueFunction):
    """
    TODO docs

    """
    output_dims = 1

    def __init__(
            self, env, model, bootstrap_model=None,
            state_action_combiner='cross'):
        self._init_combiner(state_action_combiner)
        super().__init__(env, model, bootstrap_model)

    def __call__(self, s, a):
        """
        TODO docs

        """
        X = self.X(s, a)
        Q = self.batch_eval(X)
        return Q[0]

    def X(self, s, a):
        """
        TODO docs

        """
        x = self._combiner(feature_vector(s, self.env.observation_space),
                           feature_vector(a, self.env.action_space))
        X = np.expand_dims(x, axis=0)
        return X

    def _init_combiner(self, state_action_combiner):
        self.state_action_combiner = state_action_combiner
        if state_action_combiner == 'cross':
            self._combiner = np.kron
        elif state_action_combiner == 'concatenate':
            def concat(s, a):
                return np.hstack((s, a))
            self._combiner = concat
        elif hasattr(state_action_combiner, '__call__'):
            self._combiner = state_action_combiner
        else:
            raise ValueError('bad state_action_combiner')
        assert hasattr(self, '_combiner')


class GenericQTypeII(GenericV):
    """
    TODO docs

    """
    @property
    def output_dims(self):
        if not hasattr(self, '_output_dims'):
            if not hasattr(self.env.action_space, 'n'):
                raise NonDiscreteActionSpaceError()
            self._output_dims = self.env.action_space.n
        return self._output_dims
