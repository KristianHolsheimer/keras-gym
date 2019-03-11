from .base import BaseVAlgorithm, BaseQAlgorithm, BasePolicyAlgorithm
from ..utils import ExperienceCache, accumulate_rewards


class MonteCarloV(BaseVAlgorithm):
    """
    Update the Q-function according to the plain vanilla Monte Carlo algorithm,
    cf. Section 5.3 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : state value function
        A state value function :math:`V(s)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, gamma=0.9):
        self.experience_cache = ExperienceCache(overflow='grow')
        super(MonteCarloV, self).__init__(value_function, gamma=gamma)

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
            Y = self.Y(X, A, G)     # target for function approximator

            self.value_function.update(X, Y)


class MonteCarloQ_old(BaseQAlgorithm):
    """
    Update the Q-function according to the plain vanilla Monte Carlo algorithm,
    cf. Section 5.3 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : state-action value function
        A state value function :math:`Q(s, a)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, gamma=0.9):
        self.experience_cache = ExperienceCache(overflow='grow')
        super().__init__(value_function, gamma=gamma)

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
            Y = self.Y(X, A, G)     # target for function approximator

            self.value_function.update(X, Y)


class MonteCarloQ(BaseQAlgorithm):
    """
    Update the Q-function according to the plain vanilla Monte Carlo algorithm,
    cf. Section 5.3 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : state-action value function
        A state value function :math:`Q(s, a)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, gamma=0.9):
        self.experience_cache = ExperienceCache(overflow='grow')
        super(MonteCarloQ, self).__init__(value_function, gamma=gamma)

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

        # get data from cache
        X = self.experience_cache.X_.array
        A = self.experience_cache.A_.array
        R = self.experience_cache.R_.array

        # create target
        G = accumulate_rewards(R)
        Y = self.Y(X, A, G)

        # batch update (play batch in reverse)
        self.value_function.update(X[::-1], Y[::-1])

        # clear cache for next episode
        self.experience_cache.clear()


class Reinforce(BasePolicyAlgorithm):
    def __init__(self, policy, gamma=0.9):
        self.experience_cache = ExperienceCache(overflow='grow')
        super(Reinforce, self).__init__(policy, gamma=gamma)

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

        # get data from cache
        X = self.experience_cache.X_.array
        A = self.experience_cache.A_.array
        R = self.experience_cache.R_.array

        # use (non-centered) return G as recorded advantages
        G = accumulate_rewards(R)
        advantages = G

        # batch update (play batch in reverse)
        self.policy.update(X[::-1], A[::-1], advantages[::-1])

        # clear cache for next episode
        self.experience_cache.clear()
