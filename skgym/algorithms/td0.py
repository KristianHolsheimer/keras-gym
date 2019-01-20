import numpy as np

from .base import BaseValueAlgorithm
from ..utils import idx


class BaseValueTD0(BaseValueAlgorithm):
    def update(self, s, a, r, s_next):
        X, A, R, X_next = self.preprocess_transition(s, a, r, s_next)
        Y = self.Y(X, A, R, X_next)
        self.value_function.update(X, Y)


class QLearning(BaseValueTD0):
    """
    Update the Q-function according to the Q-learning algorithm. The Q-function
    object can either be passed directly or implicitly by passing a value-based
    policy object.

    Parameters
    ----------
    value_function_or_policy : value function or value-based policy
        This can be either a state value function :math:`V(s)`, a state-action
        value function :math:`Q(s, a)`, or a value-based policy.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def target(self, X, A, R, X_next):
        Q_next = self.value_function.batch_eval_typeII(X_next)
        Q_target = R + self.gamma * np.max(Q_next, axis=1)
        return Q_target


class Sarsa(BaseValueTD0):
    """
    Update the Q-function according to the Q-learning algorithm. The Q-function
    object can either be passed directly or implicitly by passing a value-based
    policy object.

    Parameters
    ----------
    value_function_or_policy : value function or value-based policy
        This can be either a state value function :math:`V(s)`, a state-action
        value function :math:`Q(s, a)`, or a value-based policy.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def target(self, X, A, R, X_next, A_next):
        """
        Update the Q-function according to the given algorithm.

        Parameters
        ----------
        X : 2d-array, shape: [batch_size, num_features]
            Scikit-learn style design matrix. This represents a batch of either
            states or state-action pairs, depending on the model type.

        A : 1d-array, shape: [batch_size]
            A batch of actions taken.

        R : 1d-array, shape: [batch_size]
            A batch of observed rewards.

        X_next : 2d-array, shape depends on model type
            The preprocessed next-state feature vector. Its shape depends on
            the model type, if applicable. For a type-I model `X_next` has
            shape `[batch_size * num_actions, num_features]`, while a type-II
            model it has shape `[batch_size, num_features]`. Note that this
            distinction of model types only applies for value-based models. For
            a policy gradient model, `X_next` always has the shape
            `[batch_size, num_features]`.

        A_next : 1d-array, shape: [batch_size]
            A batch of 'next' actions to be taken.

        Returns
        -------
        target : 2d-array, shape: [batch_size, num_actions]
            This is the target we are optimizing towards. For instance, for a
            value-based algorithm, this is the target value for
            :math:`Q(s, a)`.

        """
        Q_next = self.value_function.batch_eval_typeII(X_next)

        # project onto next action
        Q_next = Q_next[idx(Q_next), A_next]

        # TD-target under SARSA
        Q_target = R + self.gamma * Q_next

        return Q_target

    def Y(self, X, A, R, X_next, A_next):
        """
        Given a preprocessed transition `(X, A, R, X_next)`, return the target
        to train our regressor on.

        Parameters
        ----------
        X : 2d-array, shape: [batch_size, num_features]
            Scikit-learn style design matrix. This represents a batch of either
            states or state-action pairs, depending on the model type.

        A : 1d-array, shape: [batch_size]
            A batch of actions taken.

        R : 1d-array, shape: [batch_size]
            A batch of observed rewards.

        X_next : 2d-array, shape depends on model type
            The preprocessed next-state feature vector. Its shape depends on
            the model type, if applicable. For a type-I model `X_next` has
            shape `[batch_size * num_actions, num_features]`, while a type-II
            model it has shape `[batch_size, num_features]`. Note that this
            distinction of model types only applies for value-based models. For
            a policy gradient model, `X_next` always has the shape
            `[batch_size, num_features]`.

        A_next : 1d-array, shape: [batch_size]
            A batch of 'next' actions to be taken.

        Returns
        -------
        Y : 1d- or 2d-array, depends on model type
            sklearn-style label array. Also here, the shape depends on the
            model type, if applicable. For a type-I model, the output shape is
            `[batch_size]` and for a type-II model the shape is
            `[batch_size, num_actions]`. For a policy-gradient model, the
            output shape is always `[batch_size]`.

        """
        Q_target = self.target(X, A, R, X_next, A_next)
        Y = self._Y(X, A, Q_target)
        return Y

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
        A_next = np.array([a_next])
        Y = self.Y(X, A, R, X_next, A_next)
        self.value_function.update(X, Y)
