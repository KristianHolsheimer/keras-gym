from tensorflow import keras

from ..utils import (
    is_vfunction, is_qfunction, is_policy, check_tensor, check_numpy_array)
from ..base.policy import BasePolicy
from ..base.mixins import NumActionsMixin
from ..base.function_approximators.generic import BaseFunctionApproximator


class ActorCritic(BaseFunctionApproximator, BasePolicy, NumActionsMixin):
    """
    A generic actor-critic, adorning an :term:`updateable policy` with a
    :term:`value function <state value function>`.

    The added value of using :class:`ActorCritic` to combine a policy with a
    value function is that it avoids having to feed in :term:`S` (potentially
    very large) three times at training time. Instead, it only feeds it in
    once.

    Moreover, the way :class:`ActorCritic` is implemented allows for the policy
    and value function to share parts of the computation graph, e.g. with a
    multi-head architecture.

    Parameters
    ----------
    policy : Policy object

        An :term:`updateable policy`.

    value_function : value-function object

        A :term:`state value function` :math:`V(s)`. Support for state-action
        value functions (Q-functions) is coming.

        #TODO: implement for Q-functions  -Kris

    """
    def __init__(self, policy, value_function):
        self.policy = policy
        self.value_function = value_function

        self._check_function_types()
        self._init_models()

    @property
    def env(self):
        assert self.value_function.env == self.policy.env
        return self.value_function.env

    @property
    def _cache(self):
        return self.value_function._cache

    def update(self, s, a, r, done):
        """
        Update both actor and critic.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action that was taken.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        assert self.env.observation_space.contains(s)
        assert self.env.action_space.contains(a)
        self._cache.add(s, a, r, done)

        # eager updates
        while self._cache:
            self.batch_update(*self._cache.pop())  # pop with batch_size=1

    def batch_update(self, S, A, Rn, I_next, S_next, A_next=None):
        """
        Update both actor and critic on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : 1d array, dtype: int, shape: [batch_size]

            A batch of actions that were taken.

        Rn : 1d array, dtype: float, shape: [batch_size]

            A batch of partial returns. For example, in n-step bootstrapping
            this is given by:

            .. math::

                R^{(n)}_t\\ =\\ R_t + \\gamma\\,R_{t+1} + \\dots
                    \\gamma^{n-1}\\,R_{t+n-1}

            In other words, it's the non-bootstrapped part of the n-step
            return.

        I_next : 1d array, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I_t=\\gamma^n` if the episode is ongoing
            and :math:`I_t=0` otherwise. This allows us to write the
            bootstrapped target as :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},
            A_{t+n})`.

        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 1d array, dtype: int, shape: [batch_size], optional

            A batch of next-actions that were taken. This is only required for
            SARSA (on-policy) updates.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        V_next = self.value_function.batch_eval(
            S_next,
            use_target_model=self.value_function.bootstrap_with_target_model)
        G = Rn + I_next * V_next
        check_numpy_array(G, ndim=1, dtype='float')
        check_numpy_array(A, ndim=1, dtype=('int', 'int32', 'int64'))
        losses = self._train_on_batch([S, G], [A, G])
        return losses

    def __call__(self, s):
        return self.policy(s)

    def batch_eval(self, S):
        return self.policy.batch_eval(S)

    def greedy(self, s):
        return self.policy.greedy(s)

    def proba(self, s):
        return self.policy.proba(s)

    def sync_target_model(self, tau=1.0):
        self.policy.sync_target_model(tau=tau)
        self.value_function.sync_target_model(tau=tau)

    def _check_function_types(self):
        if not is_vfunction(self.value_function):
            if is_qfunction(self.value_function):
                raise NotImplementedError(
                    "ActorCritic hasn't been yet implemented for Q-functions, "
                    "please let me know is you need this; for the time being, "
                    "please use V-function instead.")
        if not is_policy(self.policy, check_updateable=True):
            raise TypeError("expected an updateable policy")
        if self.policy.env != self.value_function.env:
            raise ValueError(
                "the envs of policy and value_function do not match")

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        # inputs
        S = keras.Input(name='actor_critic/S', shape=shape, dtype=dtype)
        G = keras.Input(name='actor_critic/G', shape=(1,), dtype='float')

        # predictions
        V = keras.layers.Lambda(self.value_function.predict_model, name='V')(S)
        Z = keras.layers.Lambda(self.policy.predict_model, name='Z')(S)
        Z_target = keras.layers.Lambda(
            self.policy.target_model, name='Z_target')(S)

        # check if shapes are what we expect
        check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(V, ndim=2, axis_size=1, axis=1)

        # update loss with advantage coming directly from graph
        policy_loss = self.policy._policy_loss(G - V, Z_target)
        value_loss = self.value_function.train_model.loss

        # joint model
        self.train_model = keras.Model(inputs=[S, G], outputs=[Z, V])
        self.train_model.compile(
            loss=[policy_loss, value_loss],
            optimizer=self.policy.train_model.optimizer)
