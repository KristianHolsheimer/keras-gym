import numpy as np

from ..base.mixins import RandomStateMixin
from ..utils import argmax


__all__ = (
    'EpsilonGreedy',
)


class EpsilonGreedy(RandomStateMixin):
    def __init__(self, q_function, epsilon=0.1, random_seed=None):
        self.q_function = q_function
        self.epsilon = epsilon
        self.random_seed = random_seed  # sets self.random in RandomStateMixin

    def __call__(self, s):
        """
        Select an action :math:`a`, given a state observation :math:`s`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a : action

            A single action.

        """
        if self.random.rand() < self.epsilon:
            return self.q_function.env.action_space.sample()

        Q = self.q_function(s)  # shape: [num_actions]
        a = argmax(Q)
        return a

    def propensity(self, s):
        Q = self.q_function(s)  # shape: [num_actions]
        a = argmax(Q)
        n = self.q_function.num_actions
        p = np.ones(n) * self.epsilon / n
        p[a] += 1 - self.epsilon
        assert p.sum() == 1
        return p
