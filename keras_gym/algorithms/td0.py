import numpy as np
from gym.spaces import Discrete

from ..utils import idx
from ..errors import NonDiscreteActionSpaceError

from .base import BaseVAlgorithm, BaseQAlgorithm


class ValueTD0(BaseVAlgorithm):
    """
    Update the state value function with TD(0) updates, cf. Section 6.1 of
    `Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_.

    Parameters
    ----------
    value_function_or_actor_critic : value function or actor-critic object

        Either a state value function :math:`V(s)` or an actor-critic object.

    gamma : float

        Future discount factor, value between 0 and 1.

    experience_cache_size : positive int, optional

        If provided, we populate a presisted experience cache that can be used
        for (asynchronous) experience replay. If left unspecified, no
        experience_cache is created. The specific value depends on your
        application. If you pick a value that's too big you might have issues
        coming from the fact early samples are less representative of the data
        generated by the current policy. Of course, there are physical
        limitations too. If you pick a value that's too small you might also
        end up with a sample that's insufficiently representative. So, the
        right value balances negative effects from remembering too much and
        forgetting too quickly.

    experience_replay_batch_size : positive int, optional

        If provided, we do experience-replay updates instead of regular, single
        instance updates.

    Attributes
    ----------
    experience_cache : ExperienceCache or None

        The persisted experience cache, which could be used for (asynchronous)
        experience-replay type updates.

    """
    def update(self, s, a, r, s_next, done):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        I_next = np.zeros(1) if done else np.array([self.gamma])

        # keep experience
        if self.experience_cache is not None:
            self.experience_cache.append(X, A, R, X_next, I_next)

        # draw from experience cache
        if self.experience_replay_batch_size:
            X, A, R, X_next, I_next = self.experience_cache.sample(
                self.experience_replay_batch_size)

        # update
        self._update_value_function_or_actor_critic(X, A, R, X_next, I_next)


class QLearning(BaseQAlgorithm):
    """
    Update the Q-function according to the Q-learning algorithm, cf.
    Section 6.5 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. The Q-function
    object can either be passed directly or implicitly by passing a value-based
    policy object.

    Parameters
    ----------
    value_function : state-action value function

        A state-action value function :math:`Q(s, a)`.

    gamma : float

        Future discount factor, value between 0 and 1.

    experience_cache_size : positive int, optional

        If provided, we populate a presisted experience cache that can be used
        for (asynchronous) experience replay. If left unspecified, no
        experience_cache is created. The specific value depends on your
        application. If you pick a value that's too big you might have issues
        coming from the fact early samples are less representative of the data
        generated by the current policy. Of course, there are physical
        limitations too. If you pick a value that's too small you might also
        end up with a sample that's insufficiently representative. So, the
        right value balances negative effects from remembering too much and
        forgetting too quickly.

    experience_replay_batch_size : positive int, optional

        If provided, we do experience-replay updates instead of regular, single
        instance updates.

    Attributes
    ----------
    experience_cache : ExperienceCache or None

        The persisted experience cache, which could be used for (asynchronous)
        experience-replay type updates.

    """
    def update(self, s, a, r, s_next, done):
        if not hasattr(self, '_update_counter'):
            self._update_counter = 0
        self._update_counter += 1

        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        I_next = np.zeros(1) if done else np.array([self.gamma])

        # keep experience
        if self.experience_cache is not None:
            self.experience_cache.append(X, A, R, X_next, I_next)

        # draw from experience cache
        if self.experience_replay_batch_size:
            if self._update_counter % self.experience_replay_batch_size == 0:
                X, A, R, X_next, I_next = self.experience_cache.sample(
                    self.experience_replay_batch_size)
            else:
                return

        # get target Q-value
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        G = R + I_next * np.max(Q_next, axis=1)  # target under Q-learning

        # update
        self._update_value_function(X, A, G)


class ExpectedSarsa(BaseQAlgorithm):
    """
    Update the Q-function according to the Expected-SARSA algorithm, cf.
    Section 6.6 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. This algorithm
    requires both a policy as well as a value function.

    Parameters
    ----------
    value_function : state-action value function

        A state-action value function :math:`Q(s, a)`.

    policy : policy object

        The policy under evaluation.

    gamma : float

        Future discount factor, value between 0 and 1.

    experience_cache_size : positive int, optional

        If provided, we populate a presisted experience cache that can be used
        for (asynchronous) experience replay. If left unspecified, no
        experience_cache is created. The specific value depends on your
        application. If you pick a value that's too big you might have issues
        coming from the fact early samples are less representative of the data
        generated by the current policy. Of course, there are physical
        limitations too. If you pick a value that's too small you might also
        end up with a sample that's insufficiently representative. So, the
        right value balances negative effects from remembering too much and
        forgetting too quickly.

    experience_replay_batch_size : positive int, optional

        If provided, we do experience-replay updates instead of regular, single
        instance updates.

    Attributes
    ----------
    experience_cache : ExperienceCache or None

        The persisted experience cache, which could be used for (asynchronous)
        experience-replay type updates.

    """
    def __init__(self, value_function, policy, gamma=0.9):
        if not isinstance(value_function.env.action_space, Discrete):
            raise NonDiscreteActionSpaceError()

        super().__init__(value_function, gamma=gamma)
        self.policy = policy

    def update(self, s, a, r, s_next, done):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        I_next = np.zeros(1) if done else np.array([self.gamma])

        # keep experience
        if self.experience_cache is not None:
            self.experience_cache.append(X, A, R, X_next, I_next)

        # draw from experience cache
        if self.experience_replay_batch_size:
            X, A, R, X_next, I_next = self.experience_cache.sample(
                self.experience_replay_batch_size)

        # get probabilities over next actions from policy
        P = self.policy.batch_eval(X_next)

        # get target Q-value
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        assert P.shape == Q_next.shape  # [batch_size, num_actions] = [b, n]
        G = R + I_next * np.einsum('bn,bn->b', P, Q_next)

        # update
        self._update_value_function(X, A, G)


class Sarsa(BaseQAlgorithm):
    """
    Update the Q-function according to the SARSA algorithm, cf.
    Section 6.4 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. The Q-function
    object can either be passed directly or implicitly by passing a value-based
    policy object.

    Parameters
    ----------
    value_function : state-action value function

        A state-action value function :math:`Q(s, a)`.

    gamma : float

        Future discount factor, value between 0 and 1.

    experience_cache_size : positive int, optional

        If provided, we populate a presisted experience cache that can be used
        for (asynchronous) experience replay. If left unspecified, no
        experience_cache is created. The specific value depends on your
        application. If you pick a value that's too big you might have issues
        coming from the fact early samples are less representative of the data
        generated by the current policy. Of course, there are physical
        limitations too. If you pick a value that's too small you might also
        end up with a sample that's insufficiently representative. So, the
        right value balances negative effects from remembering too much and
        forgetting too quickly.

    experience_replay_batch_size : positive int, optional

        If provided, we do experience-replay updates instead of regular, single
        instance updates.

    Attributes
    ----------
    experience_cache : ExperienceCache or None

        The persisted experience cache, which could be used for (asynchronous)
        experience-replay type updates.

    """
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
            A single observation (state).

        a_next : int or array
            A single action.

        """
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        I_next = np.zeros(1) if done else np.array([self.gamma])

        # keep experience
        if self.experience_cache is not None:
            self.experience_cache.append(X, A, R, X_next, I_next)

        # draw from experience cache
        if self.experience_replay_batch_size:
            X, A, R, X_next, I_next = self.experience_cache.sample(
                self.experience_replay_batch_size)

        # get target Q-value
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        Q_next = Q_next[idx(Q_next), [a_next]]  # project onto next action
        G = R + I_next * Q_next                 # TD-target under SARSA

        # update
        self._update_value_function(X, A, G)
