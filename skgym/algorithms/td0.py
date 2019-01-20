import numpy as np

from .base import BaseValueAlgorithm


class BaseValueTD0(BaseValueAlgorithm):
    def update(self, s, a, r, s_next):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        Y = self.Y(X, A, R, X_next)
        self.value_function.update(X, Y)


class QLearning(BaseValueTD0):
    """
    Update the Q-function according to the Q-learning algorithm. The Q-function
    object can either be passed directly or implicitly by passing a value-based
    policy object.

    Parameters
    ----------
    value_function_or_policy : value function or value-based policy
        This can be either a state value function :math:`V(s)`, a state-action
        value function :math:`Q(s, a)`, or a value-based policy.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def target(self, X, A, R, X_next):
        Q_next = self.value_function.batch_eval_typeII(X_next)
        Q_target = R + self.gamma * np.max(Q_next, axis=1)
        return Q_target
