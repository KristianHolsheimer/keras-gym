import numpy as np
from gym.spaces import Discrete

from .base import BaseQAlgorithm
from ..utils import ExperienceCache, idx
from ..errors import NonDiscreteActionSpaceError


class NStepQLearning(BaseQAlgorithm):
    """
    Update the Q-function according to the n-step Expected-SARSA algorithm, cf.
    Section 7.2 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. This algorithm
    requires both a policy as well as a value function.

    Parameters
    ----------
    value_function_or_policy : value function or value-based policy
        This can be either a state value function :math:`V(s)`, a state-action
        value function :math:`Q(s, a)`, or a value-based policy.

    n : int
        Number of steps to delay bootstrap estimation.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function_or_policy, n, gamma=0.9):
        super().__init__(value_function_or_policy, gamma=gamma)
        self.n = n
        self.experience_cache = ExperienceCache(maxlen=n, overflow='error')

        # private
        self._gammas = np.power(self.gamma, np.arange(self.n))

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

        # check if we need to start our updates
        if not done and len(self.experience_cache) < self.n:
            return  # wait until episode terminates or cache saturates

        # start updating if experience cache is saturated
        if not done:
            assert len(self.experience_cache) == self.n
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            Q_next = self.value_function.batch_eval_typeII(X_next)
            Q_next = np.max(Q_next, axis=1)  # the Q-learning look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)


class NStepExpectedSarsa(BaseQAlgorithm):
    """
    Update the Q-function according to the n-step Expected-SARSA algorithm, cf.
    Section 7.2 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. This algorithm
    requires both a policy as well as a value function.

    Parameters
    ----------
    value_function : value function object
        A state-action value function :math:`Q(s, a)`.

    policy : policy object
        The policy under evaluation.

    n : int
        Number of steps to delay bootstrap estimation.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, policy, n, gamma=0.9):
        if not isinstance(value_function.env.action_space, Discrete):
            raise NonDiscreteActionSpaceError()

        super().__init__(value_function, gamma=gamma)
        self.policy = policy
        self.n = n
        self.experience_cache = ExperienceCache(maxlen=n, overflow='error')

        # private
        self._gammas = np.power(self.gamma, np.arange(self.n))

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

        # check if we need to start our updates
        if not done and len(self.experience_cache) < self.n:
            return  # wait until episode terminates or cache saturates

        # start updating if experience cache is saturated
        if not done:
            assert len(self.experience_cache) == self.n
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            Q_next = self.value_function.batch_eval_typeII(X_next)
            P = self.policy.batch_eval(X_next)
            assert P.shape == Q_next.shape  # [batch_size, num_actions] = [b,n]
            Q_next = np.einsum('bn,bn->b', P, Q_next)  # exp-SARSA look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)


class NStepSarsa(BaseQAlgorithm):
    """
    Update the Q-function according to the n-step SARSA algorithm, cf.
    Section 7.2 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. This algorithm
    requires both a policy as well as a value function.

    Parameters
    ----------
    value_function_or_policy : value function or value-based policy
        This can be either a state value function :math:`V(s)`, a state-action
        value function :math:`Q(s, a)`, or a value-based policy.

    n : int
        Number of steps to delay bootstrap estimation.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function_or_policy, n, gamma=0.9):
        super().__init__(value_function_or_policy, gamma=gamma)
        self.n = n
        self.experience_cache = ExperienceCache(maxlen=n, overflow='error')

        # private
        self._gammas = np.power(self.gamma, np.arange(self.n))

    def update(self, s, a, r, s_next, a_next, done):
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
            The next state observation.

        a_next : action
            The next action.

        done : bool
            Whether the episode is done. If `done` is `False`, the input
            transition is cached and no actual update will take place. Once
            `done` is `True`, however, the collected cache from the episode is
            unrolled, replaying the epsiode in reverse chronological order.
            This is when the actual updates are made.

        """
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        self.experience_cache.append(X, A, R, X_next)

        # check if we need to start our updates
        if not done and len(self.experience_cache) < self.n:
            return  # wait until episode terminates or cache saturates

        # start updating if experience cache is saturated
        if not done:
            assert len(self.experience_cache) == self.n
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            Q_next = self.value_function.batch_eval_typeII(X_next)
            Q_next = Q_next[idx(Q_next), [a_next]]  # the SARSA look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.experience_cache.popleft_nstep(self.n)
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            Y = self.Y(X, A, G)
            self.value_function.update(X, Y)
