import numpy as np
from tensorflow import keras

from ..utils import check_numpy_array
from ..caching import NStepCache

from .base import BaseFunctionApproximator


__all__ = (
    'V',
)


class V(BaseFunctionApproximator):
    """
    A :term:`state value function` :math:`s\\mapsto v(s)`.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

    bootstrap_with_target_model : bool, optional

        Whether to use the :term:`target_model` when constructing a
        bootstrapped target. If False (default), the primary
        :term:`predict_model` is used.

    """
    def __init__(
            self, function_approximator,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False):

        self.function_approximator = function_approximator
        self.env = self.function_approximator.env
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)
        self.bootstrap_with_target_model = bool(bootstrap_with_target_model)

        self._cache = NStepCache(self.env, self.bootstrap_n, self.gamma)
        self._init_models()
        self._check_attrs()

    def __call__(self, s, use_target_model=False):
        """
        Evaluate the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        V : float or array of floats

            The estimated value of the state :math:`v(s)`.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        V = self.batch_eval(S, use_target_model=use_target_model)
        check_numpy_array(V, shape=(1,))
        V = np.asscalar(V)
        return V

    def update(self, s, r, done):
        """
        Update the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation..

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        assert self.env.observation_space.contains(s)
        self._cache.add(s, 0, r, done)

        # eager updates
        while self._cache:
            S, _, Rn, In, S_next, _ = self._cache.pop()
            self.batch_update(S, Rn, In, S_next)

    def batch_update(self, S, Rn, In, S_next):
        """
        Update the value function on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        Rn : 1d array, dtype: float, shape: [batch_size]

            A batch of partial returns. For example, in n-step bootstrapping
            this is given by:

            .. math::

                R^{(n)}_t\\ =\\ R_t + \\gamma\\,R_{t+1} + \\dots
                    \\gamma^{n-1}\\,R_{t+n-1}

            In other words, it's the non-bootstrapped part of the n-step
            return.

        In : 1d array, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I^{(n)}_t=\\gamma^n` if the episode is
            ongoing and :math:`I^{(n)}_t=0` otherwise. This allows us to write
            the bootstrapped target as:

            .. math::

                G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n}, A_{t+n})


        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        V_next = self.batch_eval(
            S_next, use_target_model=self.bootstrap_with_target_model)
        Gn = Rn + In * V_next
        losses = self._train_on_batch(S, Gn)
        return losses

    def batch_eval(self, S, use_target_model=False):
        """
        Evaluate the state value function on a batch of state observations.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        V : 1d array, dtype: float, shape: [batch_size]

            The predicted state values.

        """
        model = self.target_model if use_target_model else self.predict_model

        V = model.predict_on_batch(S)
        check_numpy_array(V, ndim=2, axis_size=1, axis=1)
        V = np.squeeze(V, axis=1)  # shape: [batch_size]
        return V

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)

        # forward pass
        X = self.function_approximator.body(S)
        V = self.function_approximator.head_v(X)

        # regular models
        self.train_model = keras.Model(S, V)
        self.train_model.compile(
            loss=self.function_approximator.VALUE_LOSS_FUNCTION,
            optimizer=self.function_approximator.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for v(s)

        # target model
        self.target_model = keras.models.clone_model(self.predict_model)
