import numpy as np
from gym.spaces import Discrete

from ..utils import idx
from ..errors import NonDiscreteActionSpaceError
from ..value_functions import GenericQ, GenericQTypeII

from .base import BaseVAlgorithm, BaseQAlgorithm


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
        G = R + self.gamma * V_next

        self.value_function.update(X, G)


class BaseQTD0Algorithm(BaseQAlgorithm):
    def preprocess_transition(self, s, a, r, s_next):
        """
        Prepare a single transition to be used for policy updates or experience
        caching.

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

        Returns
        -------
        X, A, R, X_next : arrays

            Preprocessed versions of the inputs (s, a, r, s_next).

        """
        X = self._preprocess_X(s, a)
        A = np.array([a])
        R = np.array([r])
        X_next = self.value_function.X_next(s_next)
        assert X.shape == (1, self.value_function.input_dim), "bad shape"
        assert A.shape == (1,), "bad shape"
        assert R.shape == (1,), "bad shape"

        if isinstance(self.value_function, GenericQ):
            shape = (
                1, self.value_function.num_actions,
                self.value_function.input_dim)
            assert X_next.shape == shape, "bad shape"
        elif isinstance(self.value_function, GenericQTypeII):
            shape = (1, self.value_function.input_dim)
            assert X_next.shape == shape, "bad shape"
        else:
            raise ValueError("unexpected value-function type")

        return X, A, R, X_next


class QLearning(BaseQTD0Algorithm):
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
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        G = R + self.gamma * np.max(Q_next, axis=1)  # target under Q-learning

        # update
        self._update_value_function(X, A, G)


class ExpectedSarsa(BaseQTD0Algorithm):
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

    """
    def __init__(self, value_function, policy, gamma=0.9):
        if not isinstance(value_function.env.action_space, Discrete):
            raise NonDiscreteActionSpaceError()

        super().__init__(value_function, gamma=gamma)
        self.policy = policy

    def update(self, s, a, r, s_next):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)

        # get probabilities over next actions from policy
        P = self.policy.batch_eval(X_next)

        # get target Q-value
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        assert P.shape == Q_next.shape  # [batch_size, num_actions] = [b, n]
        G = R + self.gamma * np.einsum('bn,bn->b', P, Q_next)

        # update
        self._update_value_function(X, A, G)


class Sarsa(BaseQTD0Algorithm):
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
        Q_next = self.value_function.batch_eval_next(X_next)  # bootstrap
        Q_next = Q_next[idx(Q_next), [a_next]]  # project onto next action
        G = R + self.gamma * Q_next             # TD-target under SARSA

        # update
        self._update_value_function(X, A, G)
