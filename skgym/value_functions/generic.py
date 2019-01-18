from __future__ import print_function, division
import six
import numpy as np

from .base import BaseV, BaseQ


class GenericV(BaseV):
    """
    A state value function model takes only a state as input and returns a
    single value.

    .. math::

        s\\mapsto V(s)


    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    regressor : sklearn regressor
        This regressor must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`sklearn.preprocessing.FunctionTransformer` or even
        :py:class:`sklearn.preprocessing.PolynomialFeatures`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    """
    MODELTYPE = 0

    def __call__(self, s):
        """
        Evaluate the value of a state.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        Returns
        -------
        v : float
            A single value representing :math:`V(s)`.

        """
        X_s = self.preprocess(s)
        V_s = self.batch_eval(X_s)
        return V_s[0]

    def X(self, s):
        """
        Create a feature vector from a state-action pair.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 2d-array, shape = [1, num_features]
            A sklearn-style design matrix of a single data point.

        """
        X_s = self._to_vec(s, 'state')
        X_s = np.expand_dims(X_s, axis=0)  # add batch axis (batch_size == 1)
        X_s = self._transform(X_s)  # apply transformer if provided
        return X_s

    def batch_eval(self, X_s):
        """
        Evaluate the value of a batch of states, encoded as a single
        sklearn-style design matrix.

        Parameters
        ----------
        X_s : 2d-array, shape: [batch_size, num_features]
            This a sklearn-style design matrix; it's what comes out of
            :func:`preprocess`.

        Returns
        -------
        V_s : 1d-array, shape: [batch_size]
            A batch of state values.

        """
        return self.regressor.predict(X_s)


class GenericQTypeI(BaseQ):
    """
    A type-I model takes a state-action pair as input and returns a single
    value.

    .. math::

        (s, a)\\mapsto Q(s, a)

    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    regressor : sklearn regressor
        This regressor must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`sklearn.preprocessing.FunctionTransformer` or even
        :py:class:`sklearn.preprocessing.PolynomialFeatures`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    state_action_combiner : {'cross', 'concatenate'} or func
        How to combine the feature vectors coming from `s` and `a`.
        Here 'cross' means taking a flat cross product using
        :py:func:`numpy.kron`, which gives a 1d-array of length
        `dim_s * dim_a`. This choice is suitable for simple linear models,
        including the table-lookup type models. In contrast, 'concatenate'
        uses :py:func:`numpy.hstack` to return a 1d array of length
        `dim_s + dim_a`. This option is more suitable for non-linear models.

    """
    MODELTYPE = 1

    @staticmethod
    def _concat(s, a):
        return np.hstack((s, a))

    _STATEACTION_COMBINERS = {'cross': np.kron, 'concatenate': _concat}

    def __init__(self, env, regressor, transformer=None,
                 attempt_fit_transformer=False, state_action_combiner='cross'):

        self._init_combiner(state_action_combiner)
        super(GenericQTypeI, self).__init__(
            env=env, regressor=regressor, transformer=transformer,
            attempt_fit_transformer=attempt_fit_transformer)

    def X(self, s, a):
        """
        Create a feature vector from a state-action pair. This is the design
        matrix that is fed into the regressor, i.e. function approximator. This
        is just an alias for :func:`preprocess_typeI`.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int or array
            A single action.

        Returns
        -------
        X_sa : 2d-array, shape = [1, num_features]
            A sklearn-style design matrix of a single state-action pair.

        """
        return self.preprocess_typeI(s, a)

    def preprocess_typeI(self, s, a):
        """
        Create a feature vector from a state-action pair :math:`(s,a)`.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int or array
            A single action.

        Returns
        -------
        X_sa : 2d-array, shape = [1, num_features]
            A sklearn-style design matrix of a single state-action pair.

        """
        if a is None:
            raise TypeError("'a' must be an action, got a=None")

        X_sa = self._combiner(self._to_vec(s, 'state'),
                              self._to_vec(a, 'action'))
        X_sa = np.expand_dims(X_sa, axis=0)  # add batch axis (batch_size == 1)
        X_sa = self._transform(X_sa)  # apply transformer if provided
        return X_sa

    def preprocess_typeII(self, s):
        """
        Create a feature vector from a state :math:`s`.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 3d-array, shape = [1, num_actions, num_features]
            A sklearn-style design matrix of a single action.

        """
        X_s = np.stack([self.preprocess_typeI(s, a)
                        for a in range(self.env.action_space.n)], axis=1)
        return X_s

    def batch_eval(self, X_sa):
        """
        Get a batch of values associated with a batch of state-action pairs.
        This is an alias for :func:`batch_eval_typeI`.

        Parameters
        ----------
        X_sa : 2d-array, shape: [batch_size, num_features]
            An sklearn-style design matrix representing a batch of preprocessed
            state-action pairs.

        Returns
        -------
        Q_sa : 1d-array, shape: [batch_size]
            A batch of type I Q-values.

        """
        return self.batch_eval_typeI(X_sa)

    def batch_eval_typeI(self, X_sa):
        """
        Get a batch of values associated with a batch of state-action pairs.

        Parameters
        ----------
        X_sa : 2d-array, shape: [batch_size, num_features]
            An sklearn-style design matrix representing a batch of preprocessed
            state-action pairs.

        Returns
        -------
        Q_sa : 1d-array, shape: [batch_size]
            A batch of type I Q-values.

        """
        if not X_sa.ndim == 2:
            raise TypeError(
                "bad input shape, expected a 3d array of shape: "
                "[batch_size, num_features]")

        Q_sa = self.regressor.predict(X_sa)
        assert Q_sa.ndim == 1, "expected a 1d array"
        return Q_sa

    def batch_eval_typeII(self, X_s):
        """
        Get a batch of values associated with a batch of states. This simulates
        what a type II model does. This method presumes that the actions space
        is discrete.

        Parameters
        ----------
        X_s : 3d-array, shape: [batch_size, num_actions, num_features]
            We need to supply a scikit-learn style design matrix `X_sa` for
            each possible action `a`. Here `n` is the number of possible
            actions.

        Returns
        -------
        Q_s : 2d-array, shape: [batch_size, num_actions]
            A batch of type-II Q-values.

        """
        if not hasattr(self.env.action_space, 'n'):
            raise TypeError("expected a discrete action space, type Box")
        if not X_s.ndim == 3:
            raise TypeError(
                "bad input shape, expected a 3d array of shape: "
                "[batch_size, num_actions, num_features]")

        batch_size, num_actions, num_features = X_s.shape
        if num_actions != self.env.action_space.n:
            raise TypeError(
                "bad input shape, dimension size along axis=1 must be equal "
                "to the number of actions")

        X_s = X_s.reshape([batch_size * num_actions, num_features])  # 2d
        Q_s = self.batch_eval_typeI(X_s)
        assert Q_s.shape == (batch_size * num_actions,)
        Q_s = Q_s.reshape([batch_size, num_actions])
        return Q_s

    def _init_combiner(self, state_action_combiner):
        self.state_action_combiner = state_action_combiner
        if isinstance(state_action_combiner, six.string_types):
            self._combiner = self._STATEACTION_COMBINERS[state_action_combiner]
        elif hasattr(state_action_combiner, '__call__'):
            self._combiner = state_action_combiner
        else:
            raise ValueError('bad state_action_combiner')


class GenericQTypeII(BaseQ):
    """
    A type-II model takes only a state as input and returns a vector
    (or function) defined over the space of actions.

    .. math::

        s\\mapsto Q(s, .) = [Q(s, 0), Q(s, 1), Q(s, 2), \\dots]

    In other words, a type-II model returns a value for all possible actions in
    the given state `s`.


    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    regressor : sklearn regressor
        This regressor must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`sklearn.preprocessing.FunctionTransformer` or even
        :py:class:`sklearn.preprocessing.PolynomialFeatures`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    """
    MODELTYPE = 2

    def X(self, s):
        """
        Create a feature vector from a state observation. This is the design
        matrix that is fed into the regressor, i.e. function approximator. This
        is just an alias for :func:`preprocess_typeII`.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 3d-array, shape = [1, num_actions, num_features]
            A sklearn-style design matrix of a single action.

        """
        return self.preprocess_typeII(s)

    def preprocess_typeI(self, s, a):
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
        X_sa : 2d array, shape: [1, num_features + 1]
            `X_s[:,:num_features]` is sklearn-style design matrix of a single
            state and `X_s[:,num_features]` is the corresponding array of
            actions.

        """
        X_s = self.preprocess_typeII(s)
        A = np.array([[a]], dtype=X_s.dtype)
        X_sa = np.hstack([X_s, A])
        return X_sa

    def preprocess_typeII(self, s):
        """
        Create a feature vector from a state :math:`s`.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 2d-array, shape: [1, num_features]
            A sklearn-style design matrix of a single action.

        """
        X_s = self._to_vec(s, 'state')
        X_s = np.expand_dims(X_s, axis=0)  # add batch axis (batch_size == 1)
        X_s = self._transform(X_s)  # apply transformer if provided
        return X_s

    def batch_eval(self, X_s):
        """
        Get a batch of values associated with a batch of states. This simulates
        what a type II model does. This method presumes that the actions space
        is discrete. This is an alias of :func:`batch_eval_typeII`.

        Parameters
        ----------
        X_s : 2d-array, shape: [batch_size, num_features]
            An sklearn-style design matrix representing a batch of state-action
            pairs.

        Returns
        -------
        Q_s : 2d-array, shape: [batch_size, num_actions]
            A batch of type-II Q-values.

        """
        return self.batch_eval_typeII(X_s)

    def batch_eval_typeI(self, X_sa):
        """
        Get a batch of values associated with a batch of state-action pairs.
        This method simulates what a type I model does.

        Parameters
        ----------
        X_s : 2d-array, shape: [batch_size, num_features]
            An sklearn-style design matrix representing a batch of states.

        A : 1d-array, shape = [batch_size]
            Batch of chosen actions.

        Returns
        -------
        Q_sa : 1d-array, shape: [batch_size]
            A batch of type I Q-values.

        """
        if not X_sa.ndim == 2:
            raise TypeError(
                "bad input shape, expected a 2d array of shape: "
                "[batch_size, num_features + 1]")

        # unpack X_sa -> X_s, A
        X_s = X_sa[:, :-1]
        A = X_sa[:, -1].astype('int')

        Q_s = self.batch_eval_typeII(X_s)
        idx = np.arange(Q_s.shape[0])
        Q_sa = Q_s[idx, A]

        assert Q_sa.shape == A.shape, "bad output shape"
        return Q_sa

    def batch_eval_typeII(self, X_s):
        """
        Get a batch of values associated with a batch of states. This simulates
        what a type II model does. This method presumes that the actions space
        is discrete.

        Parameters
        ----------
        X_s : 2d-array, shape: [batch_size, num_features]
            An sklearn-style design matrix representing a batch of state-action
            pairs.

        Returns
        -------
        Q_s : 2d-array, shape: [batch_size, num_actions]
            A batch of type-II Q-values.

        """
        Q_s = self.regressor.predict(X_s)
        return Q_s


class GenericQ(GenericQTypeI, GenericQTypeII):
    """
    Generic class for Q-functions. This is just a wrapper class that merges
    :class:`GenericQTypeI` and :class:`GenericQTypeII`.

    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    regressor : sklearn regressor
        This regressor must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`sklearn.preprocessing.FunctionTransformer` or even
        :py:class:`sklearn.preprocessing.PolynomialFeatures`.

    model_type : {1, 2}, optional
        Specify the model type. This is important when modeling discrete action
        spaces. A type-I model (`model_type=1`) maps
        :math:`(s,a)\\mapsto Q(s,a)`, whereas a type-II model (`model_type=2`)
        maps :math:`s\\mapsto Q(s,.)`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    state_action_combiner : {'cross', 'concatenate'} or func
        How to combine the feature vectors coming from `s` and `a`.
        Here 'cross' means taking a flat cross product using
        :py:func:`numpy.kron`, which gives a 1d-array of length
        `dim_s * dim_a`. This choice is suitable for simple linear models,
        including the table-lookup type models. In contrast, 'concatenate'
        uses :py:func:`numpy.hstack` to return a 1d array of length
        `dim_s + dim_a`. This option is more suitable for non-linear models.

    """
    _METHODS = ('preprocess', 'batch_eval_typeI', 'batch_eval_typeII')

    def __init__(self, env, regressor, transformer=None, model_type=1,
                 attempt_fit_transformer=False, state_action_combiner='cross'):

        self.model_type = self.MODELTYPE = model_type

        if self.MODELTYPE == 1:
            GenericQTypeI.__init__(
                self, env, regressor,
                transformer=transformer,
                attempt_fit_transformer=attempt_fit_transformer,
                state_action_combiner=state_action_combiner)
        elif self.MODELTYPE == 2:
            GenericQTypeII.__init__(
                self, env, regressor,
                transformer=transformer,
                attempt_fit_transformer=attempt_fit_transformer)
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")

    def __call__(self, *args):
        """
        Evaluate the value of a state-action pair.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        a : int, optional
            A single action.

        Returns
        -------
        q_sa or q_s : float or 1d-array of float, shape: [num_actions]
            Either a single float representing :math:`Q(s, a)` or a 1d array
            of floats representing :math:`Q(s, .)` if `a` is left unspecified.

        """
        if self.MODELTYPE == 1:
            return GenericQTypeI.__call__(self, *args)
        elif self.MODELTYPE == 2:
            return GenericQTypeII.__call__(self, *args)
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")

    def X(self, s, a=None):
        """
        Create a feature vector from a state :math:`s` or state-action pair
        :math:`(s, a)`, depending on the model type.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        a : int, optional
            This is required for `model_type=1` and must be left unspecified
            for `model_type=2`.

        Returns
        -------
        X : 2d array
            Scikit-learn style design matrix.

        """
        if self.MODELTYPE == 1:
            if a is None:
                raise ValueError("missing argument: 'a'")
            return GenericQTypeI.X(self, s, a)
        elif self.MODELTYPE == 2:
            if a is not None:
                raise ValueError("superfluous argument: 'a'")
            return GenericQTypeII.X(self, s)
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")

    def batch_eval_typeI(self, *args):
        """
        Get a batch of values associated with a batch of state-action pairs.

        Parameters
        ----------
        args : depends on model_type
            See :func:`skgym.value_functions.GenericQTypeI.batch_eval_typeI`
            or :func:`skgym.value_functions.GenericQTypeII.batch_eval_typeI`
            for the correct function signature.

        Returns
        -------
        Q_sa : 1d-array, shape: [batch_size]
            A batch of type I Q-values.

        """
        if self.MODELTYPE == 1:
            return GenericQTypeI.batch_eval_typeI(self, *args)
        elif self.MODELTYPE == 2:
            return GenericQTypeII.batch_eval_typeI(self, *args)
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")

    def batch_eval_typeII(self, *args):
        """
        Get a batch of values associated with a batch of states. This method
        presumes that the actions space is discrete.

        Parameters
        ----------
        args : depends on model_type
            See :func:`skgym.value_functions.GenericQTypeI.batch_eval_typeII`
            or :func:`skgym.value_functions.GenericQTypeII.batch_eval_typeII`
            for the correct function signature.

        Returns
        -------
        Q_s : 2d-array, shape: [batch_size, num_actions]
            A batch of type-II Q-values.

        """
        if self.MODELTYPE == 1:
            return GenericQTypeI.batch_eval_typeII(self, *args)
        elif self.MODELTYPE == 2:
            return GenericQTypeII.batch_eval_typeII(self, *args)
        elif self.MODELTYPE == 3:
            raise NotImplementedError("MODELTYPE == 3")
        else:
            raise ValueError("bad MODELTYPE")
