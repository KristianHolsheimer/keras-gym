import numpy as np

from .base import BaseValueAlgorithm, ExperienceCacheMixin
from ..errors import NoExperienceCacheError


class MonteCarlo(BaseValueAlgorithm, ExperienceCacheMixin):
    def target(self, R, G_next):
        """
        The emperical Monte Carlo target for our value function.

        Parameters
        ----------
        R : 1d-array, shape: [batch_size]
            Reward.

        G_next : 1d-array, shape: [batch_size]
            The next timestep's return G.

        """
        Q_target = R + self.gamma * G_next
        return Q_target

    def update(self, s, a, r, s_next, done):
        self.cache_transition(s, a, r, s_next)
        if not done:
            return

        Q_target = 0
        for X, A, R, X_next in self.replay_experience():
            Q_target = self.target(R, Q_target)
            Y = self._Y(X, A, Q_target)
            self.value_function.update(X, Y)

    def replay_experience(self):
        if not hasattr(self, 'cache_'):
            raise NoExperienceCacheError("cannot find experience cache")

        for _ in range(self.cache_['num_items']):
            X = self.cache_['X'].pop()
            A = self.cache_['A'].pop()
            R = self.cache_['R'].pop()
            X_next = self.cache_['X_next'].pop()
            self.cache_['num_items'] -= 1
            yield X, A, R, X_next
