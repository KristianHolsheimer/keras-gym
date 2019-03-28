from abc import ABC, abstractmethod
import warnings

import gym
import numpy as np
from tensorflow import keras

from ..errors import BadModelOuputShapeError, NonDiscreteActionSpaceError


class BaseValueFunction(ABC):
    """
    Abstract base class for an updateable value function.

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
    def __init__(
            self, env, model,
            target_model_sync_period=0,
            target_model_sync_tau=1.0,
            bootstrap_model=None):

        self._set_env_and_input_dim(env)

        self.model = model
        self.bootstrap_model = bootstrap_model
        self.target_model = None
        self.target_model_sync_period = int(target_model_sync_period)
        self.target_model_sync_tau = float(target_model_sync_tau)

        self._update_counter = 0
        self._init_target_model()
        self._check_model_dimensions()

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def X(self, *args):
        pass

    @abstractmethod
    def X_next(self, *args):
        pass

    def batch_eval(self, X):
        pred = self.model.predict_on_batch(X)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        return pred

    def batch_eval_next(self, X_next, use_target_model=True):
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.model
        pred = model.predict_on_batch(X_next)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        return pred

    def update(self, X, G):
        """
        Update the value function approximator.

        Parameters
        ----------
        X : 2d array, shape: [batch_size, num_features]

            A design matrix representing a batch of states or state-action
            observations.

        G : 1d array, shape: [batch_size]

            The target return, i.e. the cumulative (:math:`\\gamma`-discounted)
            rewards.

        """
        self.model.train_on_batch(X, G)
        self._check_update_target_model()

    def update_bootstrapped(self, X, Gn, X_next, I_next):
        """
        Update the value function approximator.

        This method differs form the :func:`update` method in that the
        bootstrapped target is computed on the keras/tensorflow side instead of
        the python/numpy side.

        **Note**: This method does require that a ``bootstrap_model`` is
        provided.

        Parameters
        ----------
        X : 2d array, shape: [batch_size, num_features]

            A design matrix representing a batch of states or state-action
            observations.

        Gn : 1d array, shape: [batch_size]

            The **partial** target return, i.e. the cumulative
            (:math:`\\gamma`-discounted) rewards over say :math:`n` steps (in
            the case of :math:`n`-step bootstrapping).

        X_next : 2d array, shape: [batch_size, num_features]

            Same as ``X``, except that it is used to create the bootstrapped
            target.

        I_next : 1d array, shape: [batch_size]

            The discount multiplier of the bootstrapped target, e.g.
            :math:`\\gamma^n` in the case of :math:`n`-step bootstrapping.

        """
        assert X.ndim == 2, "bad shape"
        assert X.shape[1] == self.input_dim, "bad shape"
        assert Gn.ndim == 1, "bad shape"
        assert X_next.ndim == 2, "bad shape"
        assert X_next.shape[1] == self.input_dim, "bad shape"
        assert I_next.ndim == 1, "bad shape"

        if self.bootstrap_model is None:
            raise TypeError(
                "Cannot do bootstrap updates without a `bootstrap_model`.")
        self.bootstrap_model.train_on_batch([X, X_next, I_next], Gn)

    @property
    def num_actions(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        else:
            raise NonDiscreteActionSpaceError()

    def _set_env_and_input_dim(self, env):
        self.env = env

        # create dummy X
        s = self.env.observation_space.sample()
        try:
            X = self.X(s)
        except TypeError as e:
            if "X() missing 1 required positional argument: 'a'" == e.args[0]:
                a = self.env.action_space.sample()
                X = self.X(s, a)
            else:
                raise

        # avoid overflow in model (space.sample can return very large numbers)
        X = (X - X.min()) / (X.max() - X.min())

        # set attribute
        self.input_dim = X.shape[1]

        return X

    def _check_model_dimensions(self):
        bootstrap = self.bootstrap_model is not None

        X = self._set_env_and_input_dim(self.env)
        if bootstrap:
            X_next = self._set_env_and_input_dim(self.env)
            I_next = np.ones(1)

        if self.output_dim > 1:
            G = np.zeros((1, self.output_dim))
        else:
            G = np.zeros(1)

        # check if weights can be reset
        weights_resettable = (
            hasattr(self.model, 'get_weights') and  # noqa: W504
            hasattr(self.model, 'set_weights'))

        if weights_resettable:
            weights = self.model.get_weights()

        if bootstrap:
            self.update_bootstrapped(X, G, X_next, I_next)
        try:
            self.update(X, G)
        except TypeError as e:
            msg = "update() missing 1 required positional argument: 'G'"
            if e.args[0] != msg:
                raise
            A = np.array([self.env.action_space.sample()])
            G = np.zeros(1)
            self.update(X, A, G)
        pred = self.batch_eval(X)

        if self.output_dim > 1 and pred.shape != (1, self.output_dim):
            raise BadModelOuputShapeError((1, self.output_dim), pred.shape)

        if self.output_dim == 1 and pred.shape != (1,):
            raise BadModelOuputShapeError((1,), pred.shape)

        if weights_resettable:
            self.model.set_weights(weights)

    def _init_target_model(self):
        if self.target_model_sync_period < 0:
            raise ValueError("target_model_sync_period must be non-negative")
        if not (0.0 <= self.target_model_sync_tau <= 1.0):
            raise ValueError("target_model_sync_tau must be between 0 and 1")
        if self.target_model_sync_period > 0:
            if self.bootstrap_model is not None:
                warnings.warn(
                    "target_model_sync_period > 0 is incompatible with "
                    "having a bootstrap_model; the bootstrap_model will be "
                    "dropped from the instance")
                self.bootstrap_model = None
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _check_update_target_model(self):
        self._update_counter += 1
        p = self.target_model_sync_period
        if self.target_model is not None and self._update_counter % p == 0:
            tau = self.target_model_sync_tau
            fresh_weights = self.model.get_weights()
            old_target_weights = self.target_model.get_weights()
            new_target_weights = [
                (1 - tau) * w_old + tau * w
                for w, w_old in zip(fresh_weights, old_target_weights)]
            self.target_model.set_weights(new_target_weights)
