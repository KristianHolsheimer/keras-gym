from abc import ABC, abstractmethod
import numpy as np

from ..utils import idx
from ..value_functions import GenericV, GenericQ, GenericQTypeII


class BaseAlgorithm(ABC):
    """
    Abstract base class for all algorithm objects.

    Parameters
    ----------
    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, gamma=0.9):
        self.gamma = gamma

    @abstractmethod
    def preprocess_transition(self, s, a, r, s_next):
        """
        Prepare a single transition to be used for policy updates or experience
        caching.

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

        Returns
        -------
        X, A, R, X_next : tuple of arrays
            `X` is used as input to the function approximator while `R` and
            `X_next` can be used to construct the corresponding target `Y`.

        """
        pass

    @abstractmethod
    def update(self, s, a, r, s_next, a_next=None):
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

        a_next : int or array, typically not required
            A single action. This is typicall not required, with the SARSA
            algorithm as the notable exception.

        """
        pass


class BaseVAlgorithm(BaseAlgorithm):
    """
    Abstract base class for algorithms that update a state value function
    :math:`V(s)`.

    Parameters
    ----------
    value_function : state value function
        A state value function :math:`V(s)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, gamma=0.9):
        if not isinstance(value_function, GenericV):
            raise ValueError("expected a V-type value function")
        self.value_function = value_function
        super(BaseVAlgorithm, self).__init__(gamma=gamma)

    def preprocess_transition(self, s, r, s_next):
        X = self.value_function.X(s)
        R = np.array([r])
        X_next = self.value_function.X_next(s_next)
        assert X.shape == (1, self.value_function.input_dim), "bad shape"
        assert R.shape == (1,)
        assert X_next.shape == (1, self.value_function.input_dim), "bad shape"

        return X, R, X_next


class BaseQAlgorithm(BaseAlgorithm):
    """
    Abstract base class for algorithms that update a state-action value
    function :math:`Q(s, a)`.

    Parameters
    ----------
    value_function : state-action value function
        A state-action value function :math:`Q(s, a)`.

    gamma : float
        Future discount factor, value between 0 and 1.

    """
    def __init__(self, value_function, gamma=0.9):
        if not isinstance(value_function, (GenericQ, GenericQTypeII)):
            raise ValueError("expected a Q-type value function")
        self.value_function = value_function
        super(BaseQAlgorithm, self).__init__(gamma=gamma)

    def _update_value_function(self, X, A, G):
        """ This is a little helper method to avoid duplication of code. """
        if isinstance(self.value_function, GenericQ):
            self.value_function.update(X, G)
        elif isinstance(self.value_function, GenericQTypeII):
            self.value_function.update(X, A, G)  # project onto actions
        else:
            raise ValueError("unexpected value-function type")

    def _preprocess_X(self, s, a):
        """ This is a little helper method to avoid duplication of code. """
        if isinstance(self.value_function, GenericQ):
            return self.value_function.X(s, a)
        elif isinstance(self.value_function, GenericQTypeII):
            return self.value_function.X(s)
        else:
            raise ValueError("unexpected value-function type")


class BasePolicyAlgorithm(BaseAlgorithm):
    """
    Abstract base class for algorithms that update a value function
    :math:`V(s)` or :math:`Q(s,a)`.

    Parameters
    ----------
    policy : policy object
        A policy object. Can be either a value-based policy model or a direct
        policy-gradient model.

    gamma : float
        Future discount factor, value between 0 and 1.

    TODO: write implementation, e.g. :math:`\\nabla \\log(\\pi)`

    """
    def __init__(self, policy, gamma=0.9):
        self.policy = policy
        super(BasePolicyAlgorithm, self).__init__(gamma=gamma)

    def preprocess_transition(self, s, a, r, s_next):
        """
        Prepare a single transition to be used for policy updates or experience
        caching.

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

        Returns
        -------
        X, A, R, X_next : tuple of arrays
            `X` is used as input to the function approximator while `R` and
            `X_next` can be used to construct the corresponding target `Y`.

        """
        X = self.policy.X(s)            # shape: [batch_size, num_features]
        A = np.array([a])
        R = np.array([r])
        X_next = self.policy.X(s_next)  # shape: [batch_size, num_features]

        return X, A, R, X_next
