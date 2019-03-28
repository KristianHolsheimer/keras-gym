import numpy as np

from ..utils import feature_vector

from .base import BaseValueFunction


class GenericV(BaseValueFunction):
    """
    Generic implementation of value function :math:`V(s)`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    model : keras.Model

        A Keras function approximator. The input shape can be inferred from the
        data, but the output shape must be set to ``[1]``. Check out the
        :mod:`keras_gym.wrappers` module for wrappers that allow you to use
        e.g. scikit-learn function approximators instead.

    target_model_sync_period : non-negative int, optional

        If a non-zero value is provided, the function approximator
        (:class:`keras.Model`) is copied. The copy of the model is often called
        *target* function approximator. The specific value provided for
        ``target_model_sync_period`` specifies the number of regular updates to
        perform before synchronizing the target function approximator. For
        instance, ``target_model_sync_period = 100`` means synchronize the
        target model after every 100th update of the primary model. See the
        ``target_model_sync_tau`` option below to see how the target model is
        synchronized.

    target_model_sync_tau : float, optional

        If there is a target function approximator present, this parameter
        specifies how "hard" the update must be. The update rule is:

        .. math::

            w_\\text{target}\\ \\leftarrow\\ (1-\\tau)\\,w_\\target
            + \\tau\\,w_\\text{primary}

        where :math:`w_\\text{primary}` are the weights of the primary model,
        which is continually updated. A hard update is accomplished by to the
        default value :math:`tau=1`.

    bootstrap_model : keras.Model, optional

        Just like ``model``, this is also a Keras function approximator.
        Moreover, it should use the same computation graph to the forward pass
        :math:`X(s)\\mapsto V(s)`. The way in which this model differs from the
        main ``model`` is that it takes more inputs ``[X, X_next, I_next]``
        rather than just ``X``. The additional input allows us to compute the
        bootstrapped target directly on the keras/tensorflow side, rather than
        on the python/numpy side. For a working example, have a look at the
        definition of :class:`LinearV <keras_gym.value_functions.LinearV>`.

        **Note**: Passing a ``bootstrap_model`` is completely optional. If an
        algorithm doesn't find an underlying ``bootstrap_model`` the
        bootstrapped target is computed on the python side. Also, some
        algorithms like :class:`QLearning <keras_gym.algorithms.QLearning>` are
        unable to make use of it altogether.

    Attributes
    ----------
    num_actions : int or error

        If the action space is :class:`gym.spaces.Discrete`, this is equal to
        ``env.action_space.n``. If one attempts to access this attribute when
        the action space not discrete, however, an error is raised.

    input_dim : int

        The number of input features that is fed into the function
        approximator.

    output_dim : int

        The dimensionality of the function approximator's output.

    target_model : keras.Model or None

        A copy of the underlying value function or policy. This is used to
        compute bootstrap targets. This model is typically only updated
        periodically; the period being set by the
        ``target_model_sync_period`` parameter.

    """
    output_dim = 1

    def __call__(self, s):
        """
        Get the state value :math:`V(s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        V : scalar

            This returns a scalar representing :math:`V(s)`.

        """
        X = self.X(s)
        V = self.batch_eval(X)
        return V[0]

    def X(self, s):
        """
        Get a feature vector that represents a state observation
        :math:`s`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        X : 2d array, shape: [1, num_features]

            A design matrix representing a batch (of batch_size = 1) of state
            observation(s).

        """
        x = feature_vector(s, self.env.observation_space)
        X = np.expand_dims(x, axis=0)
        return X

    def X_next(self, s_next):
        """
        Get a feature vector that represents a state observation
        :math:`s_\\text{next}`.

        This is typically used in constructing a bootstrapped target (hence the
        *_next* suffix).

        Parameters
        ----------
        s_next : state observation

            A single state observation.

        Returns
        -------
        X_next : 2d array, shape: [1, num_features]

            A design matrix representing a batch (of batch_size = 1) of state
            observation(s).

        """
        return self.X(s_next)

    def batch_eval(self, X):
        """
        Batch-evaluate the value function :math:`V(s)`.

        Parameters
        ----------
        X : 2d array, shape: [batch_size, num_features]

            A batch of feature vectors representing :math:`s`.

        Returns
        -------
        Q : 1d array, shape: [batch_size]

            This array represents :math:`V(s)`.

        """
        return super().batch_eval(X)

    def batch_eval_next(self, X_next):
        """
        Batch-evaluate the value function :math:`V(s_\\text{next})`.

        This is typically used in constructing a bootstrapped target (hence the
        *_next* suffix).

        Parameters
        ----------
        X_next : 3d array, shape: [batch_size, num_features]

            A batch of feature vectors representing :math:`s_\\text{next}`.

        Returns
        -------
        V_next : 2d array, shape: [batch_size]

            This array represents :math:`V(s)`.

        """
        return super().batch_eval_next(X_next)


class GenericQ(BaseValueFunction):
    """
    Generic implementation of value function :math:`Q(s,a)`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    model : keras.Model

        A Keras function approximator. The input shape can be inferred from the
        data, but the output shape must be set to ``[1]``. Check out the
        :mod:`keras_gym.wrappers` module for wrappers that allow you to use
        e.g. scikit-learn function approximators instead.

    state_action_combiner : {'outer', 'concatenate'} or func

        How to combine the feature vectors coming from ``s`` and ``a``. Here
        'outer' means taking a flat outer product using :py:func:`numpy.kron`,
        which gives a 1d-array of length :math:`d_s\\times d_a`. This choice is
        suitable for simple linear models, including the table-lookup type
        models. In contrast, 'concatenate' uses :py:func:`numpy.hstack` to
        return a 1d array of length :math:`d_s + d_a`. This option is more
        suitable for non-linear models.

    target_model_sync_period : non-negative int, optional

        If a non-zero value is provided, the function approximator
        (:class:`keras.Model`) is copied. The copy of the model is often called
        *target* function approximator. The specific value provided for
        ``target_model_sync_period`` specifies the number of regular updates to
        perform before synchronizing the target function approximator. For
        instance, ``target_model_sync_period = 100`` means synchronize the
        target model after every 100th update of the primary model. See the
        ``target_model_sync_tau`` option below to see how the target model is
        synchronized.

    target_model_sync_tau : float, optional

        If there is a target function approximator present, this parameter
        specifies how "hard" the update must be. The update rule is:

        .. math::

            w_\\text{target}\\ \\leftarrow\\ (1-\\tau)\\,w_\\target
            + \\tau\\,w_\\text{primary}

        where :math:`w_\\text{primary}` are the weights of the primary model,
        which is continually updated. A hard update is accomplished by to the
        default value :math:`tau=1`.

    bootstrap_model : keras.Model, optional

        Just like ``model``, this is also a Keras function approximator.
        Moreover, it should use the same computation graph to the forward pass
        :math:`X(s)\\mapsto V(s)`. The way in which this model differs from the
        main ``model`` is that it takes more inputs ``[X, X_next, I_next]``
        rather than just ``X``. The additional input allows us to compute the
        bootstrapped target directly on the keras/tensorflow side, rather than
        on the python/numpy side. For a working example, have a look at the
        definition of :class:`LinearV <keras_gym.value_functions.LinearV>`.

        **Note**: Passing a ``bootstrap_model`` is completely optional. If an
        algorithm doesn't find an underlying ``bootstrap_model`` the
        bootstrapped target is computed on the python side. Also, some
        algorithms like :class:`QLearning <keras_gym.algorithms.QLearning>` are
        unable to make use of it altogether.

    Attributes
    ----------
    num_actions : int or error

        If the action space is :class:`gym.spaces.Discrete`, this is equal to
        ``env.action_space.n``. If one attempts to access this attribute when
        the action space not discrete, however, an error is raised.

    input_dim : int

        The number of input features that is fed into the function
        approximator.

    output_dim : int

        The dimensionality of the function approximator's output.

    target_model : keras.Model or None

        A copy of the underlying value function or policy. This is used to
        compute bootstrap targets. This model is typically only updated
        periodically; the period being set by the
        ``target_model_sync_period`` parameter.

    """
    output_dim = 1

    def __init__(
            self, env, model,
            state_action_combiner='outer',
            target_model_sync_period=0,
            target_model_sync_tau=1.0,
            bootstrap_model=None):

        self._init_combiner(state_action_combiner)
        super().__init__(
            env=env,
            model=model,
            target_model_sync_period=target_model_sync_period,
            target_model_sync_tau=target_model_sync_tau,
            bootstrap_model=bootstrap_model)

    def __call__(self, s, a=None):
        """
        Get the state-action value :math:`Q(s, a)` or values
        :math:`Q(s, .)`.

        **Note**: We're only able to compute :math:`Q(s, .)` if the action
        space is discrete.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action, optional

            A single action. This argument can only be omitted if the action
            space is :class:`gym.spaces.Discrete`.

        Returns
        -------
        Q : scalar or 1d array of shape: [num_actions]

            If input ``a`` is provided, this returns a single scalar
            representing :math:`Q(s, a)`. Otherwise, this returns a vector of
            length ``num_actions`` representing :math:`Q(s, .)`.

        """
        if a is not None:
            X = self.X(s, a)
            Q = self.batch_eval(X)
        else:
            X = self.X_next(s)
            Q = self.batch_eval_next(X, use_target_model=False)
        return Q[0]

    def X(self, s, a):
        """
        Get a feature vector that represents a state-action pair
        :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action.

        Returns
        -------
        X : 2d array, shape: [1, num_features]

            A design matrix representing a batch (of batch_size = 1) of
            state-action pair(s).

        """
        x = self._combiner(feature_vector(s, self.env.observation_space),
                           feature_vector(a, self.env.action_space))
        X = np.expand_dims(x, axis=0)
        return X

    def X_next(self, s_next):
        """
        Get a feature vector that represents only a state observation
        :math:`s`.

        This is typically used in constructing a bootstrapped target (hence the
        *_next* suffix).

        **Note**: This has only been implemented for discrete action spaces.

        Parameters
        ----------
        s_next : state observation

            A single state observation. This is the state for which we will
            compute the estimated future return, i.e. bootstrapping.

        Returns
        -------
        X_next : 3d array, shape: [1, num_actions, num_features]

            A batch of feature vectors representing state observations
            :math:`s` only.

        """
        actions = np.arange(self.num_actions)  # error if space is not Discrete
        X_next = np.stack([self.X(s_next, a) for a in actions], axis=1)
        return X_next

    def batch_eval(self, X):
        """
        Batch-evaluate the value function :math:`Q(s, a)`.

        Parameters
        ----------
        X : 2d array, shape: [batch_size, num_features]

            A batch of feature vectors representing state-action pairs
            :math:`(s, a)`.

        Returns
        -------
        Q_next : 2d array, shape: [batch_size]

            This array represents :math:`Q(s, a)`.

        """
        return super().batch_eval(X)

    def batch_eval_next(self, X_next):
        """
        Evaluate the value function for all possible actions,
        :math:`Q(s, .)`.

        This is typically used in constructing a bootstrapped target (hence the
        *_next* suffix).

        **Note**: This has only been implemented for discrete action spaces.

        Parameters
        ----------
        X_next : 3d array, shape: [batch_size, num_actions, num_features]

            A batch of feature vectors representing state observations
            :math:`s` only.

        Returns
        -------
        Q_next : 2d array, shape: [batch_size, num_actions]

            This array represents :math:`Q(s, .)`.

        """
        assert X_next.ndim == 3, "bad shape"
        assert X_next.shape[1] == self.num_actions
        assert X_next.shape[2] == self.input_dim

        X_next_2d = np.reshape(X_next, [-1, self.input_dim])
        Q_next_1d = super().batch_eval_next(X_next_2d)
        assert Q_next_1d.ndim == 1, "bad shape"

        Q_next = np.reshape(Q_next_1d, [-1, self.num_actions])
        assert Q_next.shape[0] == X_next.shape[0], "bad_shape"

        return Q_next

    def _init_combiner(self, state_action_combiner):
        self.state_action_combiner = state_action_combiner
        if state_action_combiner == 'outer':
            self._combiner = np.kron
        elif state_action_combiner == 'concatenate':
            def concat(s, a):
                return np.hstack((s, a))
            self._combiner = concat
        elif hasattr(state_action_combiner, '__call__'):
            self._combiner = state_action_combiner
        else:
            raise ValueError('bad state_action_combiner')
        assert hasattr(self, '_combiner')


class GenericQTypeII(GenericV):
    """
    Generic implementation of value function :math:`Q(s,.)`.

    **Note**: This class has only been implemented for discrete action spaces.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    model : keras.Model

        A Keras function approximator. The input shape can be inferred from the
        data, but the output shape must be set to ``[1]``. Check out the
        :mod:`keras_gym.wrappers` module for wrappers that allow you to use
        e.g. scikit-learn function approximators instead.

    target_model_sync_period : non-negative int, optional

        If a non-zero value is provided, the function approximator
        (:class:`keras.Model`) is copied. The copy of the model is often called
        *target* function approximator. The specific value provided for
        ``target_model_sync_period`` specifies the number of regular updates to
        perform before synchronizing the target function approximator. For
        instance, ``target_model_sync_period = 100`` means synchronize the
        target model after every 100th update of the primary model. See the
        ``target_model_sync_tau`` option below to see how the target model is
        synchronized.

    target_model_sync_tau : float, optional

        If there is a target function approximator present, this parameter
        specifies how "hard" the update must be. The update rule is:

        .. math::

            w_\\text{target}\\ \\leftarrow\\ (1-\\tau)\\,w_\\target
            + \\tau\\,w_\\text{primary}

        where :math:`w_\\text{primary}` are the weights of the primary model,
        which is continually updated. A hard update is accomplished by to the
        default value :math:`tau=1`.

    Attributes
    ----------
    num_actions : int or error

        If the action space is :class:`gym.spaces.Discrete`, this is equal to
        ``env.action_space.n``. If one attempts to access this attribute when
        the action space not discrete, however, an error is raised.

    input_dim : int

        The number of input features that is fed into the function
        approximator.

    output_dim : int

        The dimensionality of the function approximator's output.

    target_model : keras.Model or None

        A copy of the underlying value function or policy. This is used to
        compute bootstrap targets. This model is typically only updated
        periodically; the period being set by the
        ``target_model_sync_period`` parameter.



    """
    def __init__(
            self, env, model,
            target_model_sync_period=0,
            target_model_sync_tau=1.0):

        super().__init__(
            env=env,
            model=model,
            target_model_sync_period=target_model_sync_period,
            target_model_sync_tau=target_model_sync_tau,
            bootstrap_model=None)

    @property
    def output_dim(self):
        return self.num_actions  # error if action space is not discrete

    def __call__(self, s, a=None):
        """
        Get the state-action value :math:`Q(s, a)` or values
        :math:`Q(s, .)`.

        **Note**: We're only able to compute :math:`Q(s, .)` if the action
        space is discrete.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action, optional

            A single action. This argument can only be omitted if the action
            space is :class:`gym.spaces.Discrete`.

        Returns
        -------
        Q : scalar or 1d array of shape: [num_actions]

            If input ``a`` is provided, this returns a single scalar
            representing :math:`Q(s, a)`. Otherwise, this returns a vector of
            length ``num_actions`` representing :math:`Q(s, .)`.

        """
        X = self.X(s)
        Q = self.batch_eval(X)
        assert Q.ndim == 2, "bad shape"

        if a is not None:
            # use action `a` to project Q(s, .) to Q(s, a)
            assert Q.shape[1] == self.num_actions  # error if not Discrete
            assert self.env.action_space.contains(a)
            Q = Q[:, int(a)]

        return Q[0]

    def batch_eval(self, X):
        """
        Batch-evaluate the value function :math:`Q(s, .)`.

        Parameters
        ----------
        X : 2d array, shape: [batch_size, num_features]

            A batch of feature vectors representing :math:`s`.

        Returns
        -------
        Q : 2d array, shape: [batch_size, num_actions]

            This array represents :math:`Q(s, .)`.

        """
        G_dummy = np.zeros(X.shape[0])
        pred = self.model.predict_on_batch([X, G_dummy])
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        return pred

    def batch_eval_next(self, X_next):
        """
        Batch-evaluate the value function :math:`Q(s_\\text{next}, .)`.

        This is typically used in constructing a bootstrapped target (hence the
        *_next* suffix).

        Parameters
        ----------
        X_next : 2d array, shape: [batch_size, num_features]

            A batch of feature vectors representing :math:`s_\\text{next}`.

        Returns
        -------
        Q_next : 2d array, shape: [batch_size, num_actions]

            This array represents :math:`Q(s_\\text{next}, .)`.

        """
        return super().batch_eval_next(X_next)

    def update(self, X, A, G):
        """
        Update the policy object function. This method will call
        :term:`partial_fit` on the underlying sklearn classifier.

        Parameters
        ----------
        X : 2d-array, shape = [batch_size, num_features]

            A sklearn-style design matrix of a single data point.

        A : 1d-array, shape = [batch_size]

            A batch of actions taken.

        G : 1d-array, shape = [batch_size]

            A sklearn-style label array. The shape depends on the model type.
            For a type-I model, the output shape is ``[batch_size]`` and for a
            type-II model the shape is ``[batch_size, num_actions]``.

        """
        if G.ndim == 1:
            G = np.expand_dims(G, axis=1)
        assert G.ndim == 2, "bad shape"
        assert G.shape[1] == 1, "bad shape"
        self.model.train_on_batch([X, G], A)

    def update_bootstrapped(self, *args, **kwargs):
        raise NotImplementedError('GenericQTypeII.update_bootstrapped')
    update_bootstrapped.__doc__ = None
