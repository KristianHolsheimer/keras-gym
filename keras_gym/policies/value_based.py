import numpy as np

from ..base.mixins import RandomStateMixin
from ..policies.base import BasePolicy
from ..utils import argmax


__all__ = (
    'EpsilonGreedy',
    # 'BoltzmannPolicy',  #TODO: implement
)


class EpsilonGreedy(BasePolicy, RandomStateMixin):
    """
    Value-based policy to select actions using epsilon-greedy strategy.

    Parameters
    ----------
    q_function : callable

        A state-action value function object.

    epsilon : float between 0 and 1

        The probability of selecting an action uniformly at random.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, q_function, epsilon=0.1, random_seed=None):
        self.q_function = q_function
        self.epsilon = epsilon
        self.random_seed = random_seed  # sets self.random in RandomStateMixin

    def __call__(self, s):
        if self.random.rand() < self.epsilon:
            return self.q_function.env.action_space.sample()

        a = self.greedy(s)
        return a

    def set_epsilon(self, epsilon):
        """
        Change the value for ``epsilon``.

        Parameters
        ----------
        epsilon : float between 0 and 1

            The probability of selecting an action uniformly at random.

        Returns
        -------
        self

            The updated instance.

        """
        self.epsilon = epsilon
        return self

    def greedy(self, s):
        Q = self.q_function(s)  # shape: [num_actions]
        a = argmax(Q)
        return a

    def proba(self, s):
        Q = self.q_function(s)  # shape: [num_actions]
        a = argmax(Q)
        n = self.q_function.num_actions
        p = np.ones(n) * self.epsilon / n
        p[a] += 1 - self.epsilon
        assert p.sum() == 1
        return p
