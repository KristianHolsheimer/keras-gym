from abc import ABC, abstractmethod

import gym
import numpy as np

from ..errors import BadModelOuputShapeError, NonDiscreteActionSpaceError


class BaseValueFunction(ABC):
    def __init__(self, env, model, bootstrap_model=None):
        self._set_env_and_input_dim(env)
        self.model = model
        self.bootstrap_model = bootstrap_model
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

    @abstractmethod
    def batch_eval_next(self, *args):
        pass

    def batch_eval(self, X):
        pred = self.model.predict_on_batch(X)
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

    def update_bootstrapped(self, X, Gn, X_next, I_next):
        """
        Update the value function approximator.

        This method differs form the :func:`update` method in that the
        bootstrapped target is computed on the keras/tensorflow side instead of
        the python/numpy side.

        **Note**: This method does require that a ``bootstrap_model``.

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
        assert Gn.ndim == 1, "bad shape {}".format(Gn)
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
