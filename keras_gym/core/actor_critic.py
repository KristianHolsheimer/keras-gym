from abc import abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import (
    check_numpy_array, check_tensor, is_vfunction, is_qfunction, is_policy)
from ..base.mixins import ActionSpaceMixin
from ..base.errors import ActionSpaceError
from ..policies.base import BasePolicy
from .base import BaseFunctionApproximator


__all__ = (
    'ActorCritic',
    'SoftActorCritic',
)


class BaseActorCritic(BasePolicy, BaseFunctionApproximator, ActionSpaceMixin):
    @property
    def env(self):
        return self.policy.env

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
        return self.policy(s), self.v_func(s)

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
        return self.policy.dist_params(s), self.v_func(s)

    def batch_eval(self, S, use_target_model=False):
        """
        Evaluate the actor-critic on a batch of state observations.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        A, V : arrays, shapes: [batch_size, ...] and [batch_size]

            A batch of sampled actions :term:`A` and state values :term:`V`.

        """
        A = self.policy.batch_eval(S, use_target_model=use_target_model)
        V = self.v_func.batch_eval(
            S, use_target_model=use_target_model)
        return A, V

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
        return self.policy.greedy(s), self.v_func(s)

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
        self.v_func._cache.add(s, a, r, done)

        # eager updates
        while self.v_func._cache:
            # pop with batch_size=1
            self.batch_update(*self.v_func._cache.pop())

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
        use_target_model = self.v_func.bootstrap_with_target_model
        V_next = self.v_func.batch_eval(S_next, use_target_model)
        G = Rn + In * V_next

        # check shapes / dtypes
        check_numpy_array(G, ndim=1, dtype='float')
        if self.action_space_is_discrete:
            check_numpy_array(
                A, ndim=2, dtype=('float32', 'float64'),
                axis_size=self.num_actions, axis=1)
        elif self.action_space_is_box:
            check_numpy_array(
                A, ndim=2, dtype=('float32', 'float64'),
                axis_size=self.actions_ndim, axis=1)
        else:
            raise ActionSpaceError.feature_request(self.env)

        losses = self._train_on_batch([S, A, G])
        return losses

    @abstractmethod
    def sync_target_model(self, tau=1.0):
        pass


class ActorCritic(BaseActorCritic):
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

    v_func : value-function object

        A :term:`state value function` :math:`v(s)`.

    value_loss_weight : float, optional

        Relative weight to give to the value-function loss:

        .. code:: python

            loss = policy_loss + value_loss_weight * value_loss

    """
    def __init__(self, policy, v_func, value_loss_weight=1.0):
        self.policy = policy
        self.v_func = v_func
        self.value_loss_weight = value_loss_weight

        self._check_function_types()
        self._init_models()

    def sync_target_model(self, tau=1.0):
        self.policy.sync_target_model(tau=tau)
        self.v_func.sync_target_model(tau=tau)

    def _check_function_types(self):
        if not is_vfunction(self.v_func):
            if is_qfunction(self.v_func):
                raise NotImplementedError(
                    "ActorCritic hasn't been yet implemented for Q-functions, "
                    "please let me know is you need this; for the time being, "
                    "please use V-function instead.")
        if not is_policy(self.policy, check_updateable=True):
            raise TypeError("expected an updateable policy")
        if self.policy.env != self.v_func.env:
            raise ValueError(
                "the envs of policy and v_func do not match")

    def _init_models(self):

        # inputs
        S, A = self.policy.train_model.inputs[:2]
        G = keras.Input(name='G', shape=(1,), dtype='float')

        # get TD advantages
        V = self.v_func.predict_model(S)
        Adv = K.stop_gradient(G - V)

        # update loss with advantage coming directly from graph
        policy_loss, metrics = self.policy.policy_loss_with_metrics(Adv, A)
        value_loss = self.v_func.train_model([S, G])
        metrics['policy/loss'] = policy_loss
        metrics['value/loss'] = value_loss
        loss = policy_loss + self.value_loss_weight * value_loss

        # joint model
        self.train_model = keras.Model([S, A, G], loss)
        self.train_model.add_loss(loss)
        for name, metric in metrics.items():
            self.train_model.add_metric(metric, name=name, aggregation='mean')
        self.train_model.compile(optimizer=self.policy.train_model.optimizer)


class SoftActorCritic(BaseActorCritic):
    def __init__(
            self, policy, v_func, q_func1, q_func2,
            value_loss_weight=1.0):

        self.policy = policy
        self.v_func = v_func
        self.q_func1 = q_func1
        self.q_func2 = q_func2
        self.value_loss_weight = value_loss_weight

        self._check_function_types()
        self._init_models()

    def sync_target_model(self, tau=1.0):
        self.policy.sync_target_model(tau=tau)
        self.v_func.sync_target_model(tau=tau)
        self.q_func1.sync_target_model(tau=tau)
        self.q_func2.sync_target_model(tau=tau)

    def _check_function_types(self):
        if not is_vfunction(self.v_func):
            raise TypeError("'v_func' must be a v-function: v(s)")
        if not is_qfunction(self.q_func1):
            raise TypeError("'q_func1' must be a q-function: q(s,a)")
        if not is_qfunction(self.q_func2):
            raise TypeError("'q_func2' must be a q-function: q(s,a)")
        if not is_policy(self.policy, check_updateable=True):
            raise TypeError("'policy' must be an updateable policy")
        funcs = (self.policy, self.v_func, self.q_func1, self.q_func2)
        if not all(f.env == self.env for f in funcs):
            raise ValueError(
                "the envs of policy and value function(s) do not match")

    def _init_models(self):
        # inputs
        S, A = self.policy.train_model.inputs[:2]
        G = keras.Input(name='G', shape=(1,), dtype='float')

        # get policy entropy
        H = self.policy.dist.entropy()  # differentiable

        # construct entropy-corrected target for state value function
        A_sampled = self.policy.dist.sample()  # differentiable
        log_pi = self.policy.dist.log_proba(A_sampled)
        Q1 = self.q_func1.predict_model([S, A_sampled])
        Q2 = self.q_func2.predict_model([S, A_sampled])
        check_tensor(Q1, ndim=2, axis_size=1, axis=1)
        check_tensor(Q2, same_as=Q1)
        Q_both = keras.layers.Concatenate()([Q1, Q2])
        Q_min = keras.layers.Lambda(lambda x: K.min(x, axis=1))(Q_both)
        V_target = K.stop_gradient(Q_min - self.policy.entropy_beta * log_pi)
        check_tensor(V_target, ndim=1)

        def randomly_select_qvalue(q_both):
            check_tensor(q_both, ndim=2, axis_size=2, axis=1)
            indices = K.argmax(K.random_uniform(shape=K.shape(q_both)))
            proj = K.one_hot(indices, 2)
            return tf.einsum('ij,ij->i', q_both, proj)

        # compute advantages from q-function
        V = self.v_func.predict_model(S)
        Q = keras.layers.Lambda(randomly_select_qvalue)(Q_both)
        Adv = Q - self.policy.entropy_beta * log_pi

        # update loss with advantage coming directly from graph
        policy_loss, metrics = self.policy.policy_loss_with_metrics(
            Adv=Adv, A=A_sampled)
        v_loss = self.v_func.train_model([S, V_target])
        q_loss1 = self.q_func1.train_model([S, A, G])
        q_loss2 = self.q_func2.train_model([S, A, G])
        value_loss = (v_loss + q_loss1 + q_loss2) / 3.

        # add losses to metrics dict
        metrics.update({
            'policy/loss': policy_loss,
            'policy/entropy': H,
            'v_func/loss': v_loss,
            'q_func1/loss': q_loss1,
            'q_func2/loss': q_loss2,
            'value/loss': value_loss,
        })

        # combined loss function
        loss = policy_loss + self.value_loss_weight * value_loss
        check_tensor(loss, ndim=0)  # should be a scalar

        # joint model
        self.train_model = keras.Model([S, A, G], loss)
        self.train_model.add_loss(loss)
        for name, metric in metrics.items():
            self.train_model.add_metric(metric, name=name, aggregation='mean')
        self.train_model.compile(optimizer=self.policy.train_model.optimizer)
