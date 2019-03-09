import numpy as np
from gym.spaces import Discrete

from .base import BaseVAlgorithm, BaseQAlgorithm
from ..utils import idx
from ..errors import NonDiscreteActionSpaceError


class ValueTD0(BaseVAlgorithm):
    """
    Update the state value function with TD(0) updates, cf. Section 6.1 of
    `Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_. The
    Q-function object can either be passed directly or implicitly by passing a
    value-based policy object.

    Parameters
    ----------
    value_function : value function object
        A state-action value function :math:`Q(s, a)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def update(self, s, r, s_next):
        X, R, X_next = self.preprocess_transition(s, r, s_next)

        # get TD target
        V_next = self.value_function.batch_eval(X_next)  # bootstrap
        Y = R + self.gamma * V_next

        self.value_function.update(X, Y)


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

    """
    def update(self, s, a, r, s_next):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)

        # get target Q-value
        Q_next = self.value_function.batch_eval_typeII(X_next)  # bootstrap
        G = R + self.gamma * np.max(Q_next, axis=1)  # target under Q-learning

        # target for function approximator
        Y = self.Y(X, A, G)
        self.value_function.update(X, Y)


class ExpectedSarsa(BaseQAlgorithm):
    """
    Update the Q-function according to the Expected-SARSA algorithm, cf.
    Section 6.6 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. This algorithm
    requires both a policy as well as a value function.

    # FIXME: Use proper policy to compute action propensities. Perhaps just
    epsilon greedy propensities.

    Parameters
    ----------
    value_function : state-action value function
        A state-action value function :math:`Q(s, a)`.

    policy : policy object
        The policy under evaluation.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, policy, gamma=0.9):
        if not isinstance(value_function.env.action_space, Discrete):
            raise NonDiscreteActionSpaceError()

        super(ExpectedSarsa, self).__init__(value_function, gamma=gamma)
        self.policy = policy

    def update(self, s, a, r, s_next):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)

        # get probabilities over next actions from policy
        P = self.policy.batch_eval(X_next)

        # get target Q-value
        Q_next = self.value_function.batch_eval_typeII(X_next)  # bootstrap
        assert P.shape == Q_next.shape  # [batch_size, num_actions] = [b, n]
        G = R + self.gamma * np.einsum('bn,bn->b', P, Q_next)

        # target for function approximator
        Y = self.Y(X, A, G)
        self.value_function.update(X, Y)


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

    """
    def update(self, s, a, r, s_next, a_next):
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

        # get target Q-value
        Q_next = self.value_function.batch_eval_typeII(X_next)  # bootstrap
        Q_next = Q_next[idx(Q_next), [a_next]]  # project onto next action
        G = R + self.gamma * Q_next             # TD-target under SARSA

        # target for function approximator
        Y = self.Y(X, A, G)
        self.value_function.update(X, Y)
