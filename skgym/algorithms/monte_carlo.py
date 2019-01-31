from .base import BaseValueAlgorithm
from ..utils import ExperienceCache


class MonteCarlo(BaseValueAlgorithm):
    def __init__(self, value_function_or_policy, gamma=0.9):
        self.experience_cache = ExperienceCache(overflow='grow')
        super(MonteCarlo, self).__init__(value_function_or_policy, gamma=gamma)

    def target(self, *args, **kwargs):
        raise NotImplementedError('MonteCarlo.target')

    def update(self, s, a, r, s_next, done):
        """
        Update the given policy and/or value function.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int or array
            A single action.

        r : float
            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

        s_next : int or array
            A single observation (state).

        done : bool
            Whether the episode is done. If `done` is `False`, the input
            transition is cached and no actual update will take place. Once
            `done` is `True`, however, the collected cache from the episode is
            unrolled, replaying the epsiode in reverse chronological order.
            This is when the actual updates are made.

        """
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        self.experience_cache.append(X, A, R, X_next)

        # break out of function if episode hasn't yet finished
        if not done:
            return

        # initialize return
        G = 0

        # replay episode in reverse order
        while self.experience_cache:
            X, A, R, X_next = self.experience_cache.pop()

            G = R + self.gamma * G  # gamma-discounted return
            Y = self._Y(X, A, G)    # target for function approximator

            self.value_function.update(X, Y)
