from abc import ABC, abstractmethod

import numpy as np

from ..errors import NonDiscreteActionSpaceError


class BaseValueFunction(ABC):
    """
    Abstract base class for value functions.

    MODELTYPES
    ----------
    0 : s     -->  V(s)
    1 : s, a  -->  Q(s, a)  for either discrete or continuous action spaces
    2 : s     -->  Q(s, .)  for discrete action spaces
    3 : s     -->  Q(s, .)  for continuous action spaces (not yet implemented)

    TODO
    ----
    add batch_eval_typeIII and model type 3

    """
    MODELTYPES = (0, 1, 2)

    def __init__(self, env, model):
        self.env = env
        self.model = model
        self._check_model()

    @abstractmethod
    def __call__(self, *args):
        """
        Compute the value for a state observation or state-action pair
        (depending on model type).

        Parameters
        ----------
        args
            Either state or state-action pair, depending on model type.

        Returns
        -------
        v, q_I, q_II or q_III : float, float, array of floats, or func
            A sklearn-style design matrix of a single data point. For a state
            value function (type 0) as well as for a type I model this returns
            a single float. For a type II model this returns an array of
            Q-values. For a type III model, this returns a callable object
            (function) that maps :math:`a\\mapsto Q(s,a)`.

        """
        pass

    @abstractmethod
    def X(self, *args):
        """
        Create a feature vector from a state observation or state-action pair.
        This is the design matrix that is fed into the Keras model, i.e.
        function approximator.

        Parameters
        ----------
        args
            Either state or state-action pair, depending on model type.

        Returns
        -------
        X : 2d-array, shape = [1, num_features]
            A sklearn-style design matrix of a single data point.

        """
        pass

    def update(self, X, Y):
        """
        Update the value function. This method will call the `train_on_batch`
        method on the underlying Keras model, i.e. function approximator.

        Parameters
        ----------
        X : 2d-array, shape = [batch_size, num_features]
            A sklearn-style design matrix of a single data point.

        Y : 1d- or 2d-array, depends on model type
            A sklearn-style label array. The shape depends on the model type.
            For a type-I model, the output shape is `[batch_size]` and for a
            type-II model the shape is `[batch_size, num_actions]`.

        """
        self.model.train_on_batch(X, Y)

    def _create_dummy_X_Y(self):
        # n is needed to create dummy output Y
        try:
            n = self.env.action_space.n
        except AttributeError:
            raise NonDiscreteActionSpaceError()

        # sample a state observation from the environment
        s = self.env.observation_space.sample()
        if isinstance(s, np.ndarray):
            s = np.random.rand(*s.shape)  # otherwise we get overflow

        if self.MODELTYPE == 0:
            X = self.X(s)
            Y = np.zeros(1)
        elif self.MODELTYPE == 1:
            a = self.env.action_space.sample()
            X = self.X(s, a)
            Y = np.zeros(1)
        elif self.MODELTYPE == 2:
            X = self.X(s)
            Y = np.zeros((1, n))
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")

        # set some attributes for convenience
        # N.B. value_functions.predefined.Linear{V,Q} require these to be set
        self.num_features = X.shape[1]
        self.num_actions = n

        return X, Y

    def _check_model(self):
        # get some dummy data
        X, Y = self._create_dummy_X_Y()

        weights_resettable = (
            hasattr(self.model, 'get_weights') and  # noqa: W504
            hasattr(self.model, 'set_weights'))

        if weights_resettable:
            weights = self.model.get_weights()

        try:
            self.model.train_on_batch(X, Y)
            pred = self.batch_eval(X)
            if self.MODELTYPE in (0, 1):
                assert pred.shape == (1,), "bad shape"
            elif self.MODELTYPE == 2:
                assert pred.shape == (1, self.num_actions), "bad shape"
            elif self.MODELTYPE == 3:
                # num_params = ...  # params distr over continuous actions
                # assert pred.shape == (num_params,), "bad model output shape"
                raise NotImplementedError("MODELTYPE == 3")
        except Exception as e:
            # TODO: show informative error message
            raise e

        if weights_resettable:
            self.model.set_weights(weights)


class BaseV(BaseValueFunction):
    MODELTYPE = 0

    @abstractmethod
    def batch_eval(self, X):
        pass


class BaseQ(BaseValueFunction):
    MODELTYPES = (1, 2)

    def __call__(self, s, a=None):
        """
        Evaluate the value of a state-action pair.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int or array, optional
            A single action.

        Returns
        -------
        q_sa or q_s : float or 1d-array of float, shape: [num_actions]
            Either a single float representing :math:`Q(s, a)` or a 1d array
            of floats representing :math:`Q(s, .)` if `a` is left unspecified.

        """
        if a is None:
            X_s = self.preprocess_typeII(s)
            Q_s = self.batch_eval_typeII(X_s)
            return Q_s[0]
        else:
            X_sa = self.preprocess_typeI(s, a)
            Q_sa = self.batch_eval_typeI(X_sa)
            return Q_sa[0]

    @abstractmethod
    def preprocess_typeI(self, *args):
        """
        Create a feature vector from a state-action pair.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int
            A single action.

        Returns
        -------
        X_sa : 2d array
            A sklearn-style design matrix of a single action. The shape depends
            on the model type.

        """
        pass

    @abstractmethod
    def preprocess_typeII(self, s):
        """
        Create a feature vector from a state :math:`s`.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 2d-array
            A sklearn-style design matrix of a single action. The shape depends
            on the model type.

        """
        pass

    @abstractmethod
    def batch_eval_typeI(self, *args):
        pass

    @abstractmethod
    def batch_eval_typeII(self, *args):
        pass
