from tensorflow import keras

from .. import utils
from ..base.mixins import ActionSpaceMixin
from ..base.errors import ActionSpaceError
from ..policies.base import BasePolicy
from .base import BaseFunctionApproximator


__all__ = (
    'ActorCritic',
)


class ActorCritic(BasePolicy, BaseFunctionApproximator, ActionSpaceMixin):
    """
    A generic actor-critic, combining an :term:`updateable policy` with a
    :term:`value function <state value function>`.

    The added value of using an :class:`ActorCritic` to combine a policy with a
    value function is that it avoids having to feed in :term:`S` (potentially
    very large) three times at training time. Instead, it only feeds it in
    once.

    Parameters
    ----------
    policy : Policy object

        An :term:`updateable policy`.

    value_function : value-function object

        A :term:`state value function` :math:`v(s)`. Support for state-action
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

            A single action.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        assert self.env.observation_space.contains(s)
        self._cache.add(s, a, r, done)

        # eager updates
        while self._cache:
            self.batch_update(*self._cache.pop())  # pop with batch_size=1

    def batch_update(self, S, A, Rn, In, S_next, A_next=None):
        """
        Update both actor and critic on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : nd Tensor, shape: [batch_size, ...]

            A batch of actions taken.

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
            the bootstrapped target as
            :math:`G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n}, A_{t+n})`.

        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 2d Tensor, shape: [batch_size, ...]

            A batch of (potential) next actions :term:`A_next`. This argument
            is only used if ``update_strategy='sarsa'``.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        use_target_model = self.value_function.bootstrap_with_target_model
        V_next = self.value_function.batch_eval(S_next, use_target_model)
        G = Rn + In * V_next

        # check shapes / dtypes
        utils.check_numpy_array(G, ndim=1, dtype='float')
        if self.action_space_is_discrete:
            utils.check_numpy_array(
                A, ndim=1, dtype='int')
        elif self.action_space_is_box:
            utils.check_numpy_array(
                A, ndim=2, dtype='float', axis_size=self.actions_ndim, axis=1)
        else:
            raise ActionSpaceError.feature_request(self.env)

        losses = self._train_on_batch([S, G], [A, G])  # FIXME: change signature
        return losses

    def __call__(self, s):
        """
        Draw an action from the current policy :math:`\\pi(a|s)` and get the
        expected value :math:`v(s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a, v : tuple (1d array of floats, float)

            Returns a pair representing :math:`(a, v(s))`.

        """
        return self.policy(s), self.value_function(s)

    def dist_params(self, s):
        """

        Get the distribution parameters under the current policy
        :math:`\\pi(a|s)` and get the expected value :math:`v(s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        dist_params, v : tuple (1d array of floats, float)

            Returns a pair representing the distribution parameters of
            :math:`\\pi(a|s)` and the estimated state value :math:`v(s)`.

        """
        return self.policy.dist_params(s), self.value_function(s)

    def greedy(self, s):
        """
        Draw a greedy action :math:`a=\\arg\\max_{a'}\\pi(a'|s)` and get the
        expected value :math:`v(s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a, v : tuple (1d array of floats, float)

            Returns a pair representing :math:`(a, v(s))`.

        """
        return self.policy.greedy(s), self.value_function(s)

    def batch_eval(self, S, use_target_model=False):
        """
        Evaluate the policy on a batch of state observations.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        V, dist : (array, ProbaDist with shapes: ([batch_size, num_actions], [batch_size])

            A batch of action probabilities and values
            :math:`(\\pi(.|s), v(s))`.

        """  # noqa: E501
        P = self.policy.batch_eval(S, use_target_model=use_target_model)
        V = self.value_function.batch_eval(
            S, use_target_model=use_target_model)
        return P, V

    def sync_target_model(self, tau=1.0):
        self.policy.sync_target_model(tau=tau)
        self.value_function.sync_target_model(tau=tau)

    def _check_function_types(self):
        if not utils.is_vfunction(self.value_function):
            if utils.is_qfunction(self.value_function):
                raise NotImplementedError(
                    "ActorCritic hasn't been yet implemented for Q-functions, "
                    "please let me know is you need this; for the time being, "
                    "please use V-function instead.")
        if not utils.is_policy(self.policy, check_updateable=True):
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
        V = self.value_function.predict_model(S)
        Z = self.policy.predict_model(S)
        Z_target = keras.layers.Lambda(
            self.policy.target_model, name='Z_target')(S)

        # check if shapes are what we expect
        if self.action_space_is_discrete:
            utils.check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        elif self.action_space_is_box:
            utils.check_tensor(Z, ndim=3, axis_size=self.actions_ndim, axis=1)
            utils.check_tensor(Z, axis_size=2, axis=2)
        utils.check_tensor(V, ndim=2, axis_size=1, axis=1)

        # update loss with advantage coming directly from graph
        policy_loss = self.policy._policy_loss(G - V, Z_target)
        value_loss = self.value_function.train_model.loss

        # joint model
        self.train_model = keras.Model(inputs=[S, G], outputs=[Z, V])
        self.train_model.compile(
            loss=[policy_loss, value_loss],
            optimizer=self.policy.train_model.optimizer)
