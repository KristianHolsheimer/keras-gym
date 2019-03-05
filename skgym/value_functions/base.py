import sys
from abc import ABC, abstractmethod

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError


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

    def __init__(self, env, regressor, transformer=None,
                 attempt_fit_transformer=False):
        self.env = env
        self.regressor = regressor
        self.transformer = transformer
        self.attempt_fit_transformer = attempt_fit_transformer
        self._init_model()

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
        This is the design matrix that is fed into the regressor, i.e. function
        approximator.

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
        Update the value function. This method will call :term:`partial_fit` on
        the underlying sklearn regressor.

        Parameters
        ----------
        X : 2d-array, shape = [batch_size, num_features]
            A sklearn-style design matrix of a single data point.

        Y : 1d- or 2d-array, depends on model type
            A sklearn-style label array. The shape depends on the model type.
            For a type-I model, the output shape is `[batch_size]` and for a
            type-II model the shape is `[batch_size, num_actions]`.

        """
        self.regressor.partial_fit(X, Y)

    def _init_model(self):
        # n is needed to create dummy output Y
        try:
            n = self.env.action_space.n
        except AttributeError:
            raise NotImplementedError(
                "can only do discrete action spaces for now")

        # create dummy input X
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

        try:
            self.regressor.partial_fit(X, Y)
        except ValueError as e:
            expected_failure = (
                e.args[0].startswith("bad input shape") and  # Y has bad shape
                self.MODELTYPE == 2 and                      # type II model
                not isinstance(
                    self.regressor, MultiOutputRegressor))   # not yet wrapped
            if not expected_failure:
                raise
            self.regressor = MultiOutputRegressor(self.regressor)
            self.regressor.partial_fit(X, Y)

    def _transform(self, X):
        if self.transformer is not None:
            try:
                X = self.transformer.transform(X)
            except NotFittedError:
                if not self.attempt_fit_transformer:
                    raise NotFittedError(
                        "transformer needs to be fitted; setting "
                        "attempt_fit_transformer=True will fit the "
                        "transformer on one data point")
                print("attemting to fit transformer", file=sys.stderr)
                X = self.transformer.fit_transform(X)
        return X


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
