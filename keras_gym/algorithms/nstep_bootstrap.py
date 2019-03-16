import numpy as np
from gym.spaces import Discrete

from ..utils import ExperienceCache
from ..errors import NonDiscreteActionSpaceError

from .base import BaseVAlgorithm
from .td0 import BaseQTD0Algorithm


class NStepBootstrapV(BaseVAlgorithm):
    """
    Update state value function according to the n-step bootstrap TD.

    See Section 7.1 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : value function

        A state-action value function :math:`V(s)`.

    n : int

        Number of steps to delay bootstrap estimation.

    gamma : float

        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, n, gamma=0.9):
        super().__init__(value_function, gamma=gamma)
        self.n = n
        self.experience_cache = ExperienceCache(maxlen=n, overflow='error')

        # private
        self._gammas = np.power(self.gamma, np.arange(self.n))

    def preprocess_transition(self, s, r, s_next):
        """
        Prepare a single transition to be used for policy updates or experience
        caching.

        Parameters
        ----------
        s : int or array

            A single observation (state).

        r : float

            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

        s_next : state observation

            A single state observation. This is the state for which we will
            compute the estimated future return, i.e. bootstrapping.

        Returns
        -------
        X, R, X_next : arrays

            Preprocessed versions of the inputs (s, r, s_next).

        """
        X = self.value_function.X(s)
        R = np.array([r])
        X_next = self.value_function.X_next(s_next)
        assert X.shape == (1, self.value_function.input_dim), "bad shape"
        assert R.shape == (1,), "bad shape"
        assert X_next.shape == (1, self.value_function.input_dim), "bad shape"

        return X, R, X_next

    def popleft_nstep(self):
        """
        Pop the oldest cached transition and return the `R` and `X_next` that
        correspond to an n-step look-ahead.

        **Note:** To understand of what's going in this method, have a look at
        chapter 7 of `Sutton & Barto
        <http://incompleteideas.net/book/the-book-2nd.html>`_.

        Returns
        -------
        X, R, X_next : arrays

            A batch of preprocessed transitions. ``X`` corresponds to the
            to-be-updated timestep :math:`\\tau=t-n+1`, while ``X_next``
            corresponds to the look-ahead timestep :math:`\\tau+n=t+1`. ``R``
            contains all the observed rewards between timestep :math:`\\tau+1`
            and :math:`\\tau+n` (inclusive), i.e. ``R`` represents the sequence
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{\\tau+n})`. This sequence
            is truncated to a size smaller than :math:`n` as we approach the
            end of the episode, where :math:`t>T-n`. The sequence becomes
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{T})`. In this phase of the
            replay, we can longer do a bootstrapping type look-ahead, which
            means that ``X_next=None`` until the end of the episode.

        """
        self._check_fitted()
        c = self.experience_c
        n = self.n

        X = np.expand_dims(c.deques_[0].popleft(), axis=0)

        R = c.deques_[1].array[:n]
        c.deques_[1].popleft()

        X_next = c.deques_[2].array[[-1]] if len(c.deques_[2]) >= n else None
        c.deques_[2].popleft()

        return X, R, X_next

    def update(self, s, r, s_next, done):
        """
        Update the given value function.

        Parameters
        ----------
        s : int or array

            A single observation (state).

        r : float

            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

        s_next : state observation

            A single state observation. This is the state for which we will
            compute the estimated future return, i.e. bootstrapping.

        done : bool

            Whether the episode is done. If ``done`` is ``False``, the input
            transition is cached and no actual update will take place. Once
            ``done`` is ``True``, however, the collected cache from the episode
            is unrolled, replaying the epsiode in reverse chronological order.
            This is when the actual updates are made.

        """
        X, R, X_next = self.preprocess_transition(s, r, s_next)
        self.experience_cache.append(X, R, X_next)

        # check if we need to start our updates
        if not done and len(self.experience_cache) < self.n:
            return  # wait until episode terminates or cache saturates

        # start updating if experience cache is saturated
        if not done:
            assert len(self.experience_cache) == self.n
            X, R, X_next = self.popleft_nstep()
            Q_next = self.value_function.batch_eval_next(X_next)
            Q_next = np.max(Q_next, axis=1)  # the Q-learning look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            self.value_function.update(X, G)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, R, X_next = self.popleft_nstep()
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            self.value_function.update(X, G)


class BaseNStepQAlgorithm(BaseQTD0Algorithm):
    """ inherit preprocess_trasition from BaseQTD0Algorithm """
    def __init__(self, value_function, n, gamma=0.9):
        super().__init__(value_function, gamma=gamma)
        self.n = n
        self.experience_cache = ExperienceCache(maxlen=n, overflow='error')

        # private
        self._gammas = np.power(self.gamma, np.arange(self.n))

    def popleft_nstep(self):
        """
        Pop the oldest cached transition and return the `R` and `X_next` that
        correspond to an n-step look-ahead.

        **Note:** To understand of what's going in this method, have a look at
        chapter 7 of `Sutton & Barto
        <http://incompleteideas.net/book/the-book-2nd.html>`_.

        Returns
        -------
        X, A, R, X_next : arrays

            A batch of preprocessed transitions. ``X`` and ``A`` correspond to
            the to-be-updated timestep :math:`\\tau=t-n+1`, while ``X_next``
            corresponds to the look-ahead timestep :math:`\\tau+n=t+1`. ``R``
            contains all the observed rewards between timestep :math:`\\tau+1`
            and :math:`\\tau+n` (inclusive), i.e. ``R`` represents the sequence
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{\\tau+n})`. This sequence
            is truncated to a size smaller than :math:`n` as we approach the
            end of the episode, where :math:`t>T-n`. The sequence becomes
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{T})`. In this phase of the
            replay, we can longer do a bootstrapping type look-ahead, which
            means that ``X_next=None`` until the end of the episode.

        """
        c = self.experience_cache
        c._check_fitted()
        n = self.n

        X = np.expand_dims(c.deques_[0].popleft(), axis=0)
        A = np.expand_dims(c.deques_[1].popleft(), axis=0)

        R = c.deques_[2].array[:n]
        c.deques_[2].popleft()

        X_next = c.deques_[3].array[[-1]] if len(c.deques_[3]) >= n else None
        c.deques_[3].popleft()

        return X, A, R, X_next


class NStepQLearning(BaseNStepQAlgorithm):
    """
    Update the Q-function according to the n-step Expected-SARSA algorithm.

    See Section 7.2 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : value function

        A state-action value function :math:`Q(s, a)`.

    n : int

        Number of steps to delay bootstrap estimation.

    gamma : float

        Future discount factor, value between 0 and 1.

    """
    def update(self, s, a, r, s_next, done):
        """
        Update the given value function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action a

        r : float

            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

        s_next : state observation

            A single state observation. This is the state for which we will
            compute the estimated future return, i.e. bootstrapping.

        done : bool

            Whether the episode is done. If ``done`` is ``False``, the input
            transition is cached and no actual update will take place. Once
            ``done`` is ``True``, however, the collected cache from the episode
            is unrolled, replaying the epsiode in reverse chronological order.
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
            X, A, R, X_next = self.popleft_nstep()
            Q_next = self.value_function.batch_eval_next(X_next)
            Q_next = np.max(Q_next, axis=1)  # the Q-learning look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            self._update_value_function(X, A, G)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.popleft_nstep()
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            self._update_value_function(X, A, G)


class NStepExpectedSarsa(BaseNStepQAlgorithm):
    """
    Update the Q-function according to the n-step Expected-SARSA algorithm.

    See Section 7.2 of `Sutton & Barto
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
        super().__init__(value_function, n, gamma)
        self.policy = policy

    def update(self, s, a, r, s_next, done):
        """
        Update the given value function.

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
            X, A, R, X_next = self.popleft_nstep()
            Q_next = self.value_function.batch_eval_next(X_next)
            P = self.policy.batch_eval(X_next)
            assert P.shape == Q_next.shape  # [batch_size, num_actions] = [b,n]
            Q_next = np.einsum('bn,bn->b', P, Q_next)  # exp-SARSA look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            self._update_value_function(X, A, G)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.popleft_nstep()
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            self._update_value_function(X, A, G)


class NStepSarsa(BaseNStepQAlgorithm):
    """
    Update the Q-function according to the n-step SARSA algorithm.

    See Section 7.2 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function : value function

        A state-action value function :math:`Q(s, a)`.

    n : int

        Number of steps to delay bootstrap estimation.

    gamma : float

        Future discount factor, value between 0 and 1.

    """
    def update(self, s, a, r, s_next, a_next, done):
        """
        Update the given value function.

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
            X, A, R, X_next = self.popleft_nstep()
            Q_next = self.value_function.batch_eval_next(X_next)
            Q_next = Q_next[[0], [a_next]]  # the SARSA look-ahead
            G = self._gammas.dot(R) + np.power(self.gamma, self.n + 1) * Q_next
            self._update_value_function(X, A, G)

            return  # wait until episode terminates

        # roll out remainder of episode
        while self.experience_cache:
            X, A, R, X_next = self.popleft_nstep()
            G = np.expand_dims(self._gammas[:len(R)].dot(R), axis=0)
            self._update_value_function(X, A, G)
