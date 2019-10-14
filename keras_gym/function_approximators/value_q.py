from abc import abstractmethod

import numpy as np
from tensorflow import keras

from ..base.errors import ActionSpaceError
from ..utils import one_hot, check_numpy_array, project_onto_actions_np
from ..caching import NStepCache
from ..losses import ProjectedSemiGradientLoss

from .base import BaseFunctionApproximator


__all__ = (
    'QTypeI',
    'QTypeII',
)


class BaseQ(BaseFunctionApproximator):
    UPDATE_STRATEGIES = ('sarsa', 'q_learning', 'double_q_learning')

    def __init__(
            self, function_approximator,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='sarsa'):

        self.function_approximator = function_approximator

        self.env = self.function_approximator.env
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)
        self.bootstrap_with_target_model = bool(bootstrap_with_target_model)
        self.update_strategy = update_strategy

        self._cache = NStepCache(self.env, self.bootstrap_n, self.gamma)
        self._init_models()
        self._check_attrs()

    def __call__(self, s, a=None, use_target_model=False):
        """
        Evaluate the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action, optional

            A single action.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        Q : float or array of floats

            If action ``a`` is provided, a single float representing
            :math:`q(s,a)` is returned. If, on the other hand, ``a`` is left
            unspecified, a vector representing :math:`q(s,.)` is returned
            instead. The shape of the latter return value is ``[num_actions]``,
            which is only well-defined for discrete action spaces.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        if a is not None:
            assert self.env.action_space.contains(a)
            if self.action_space_is_discrete:
                a = self._one_hot_encode_discrete(a)
            A = np.expand_dims(a, axis=0)
            Q = self.batch_eval(S, A, use_target_model=use_target_model)
            check_numpy_array(Q, shape=(1,))
            Q = np.asscalar(Q)
        else:
            Q = self.batch_eval(S, use_target_model=use_target_model)
            check_numpy_array(Q, shape=(1, self.num_actions))
            Q = np.squeeze(Q, axis=0)
        return Q

    def update(self, s, a, r, done):
        """
        Update the Q-function.

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

    @abstractmethod
    def batch_update(self, S, A, Rn, In, S_next, A_next=None):
        """
        Update the value function on a batch of transitions.

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
            the bootstrapped target as:

            .. math::

                G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n}, A_{t+n})


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
        pass

    def bootstrap_target(self, Rn, In, S_next, A_next=None):
        """
        Get the bootstrapped target
        :math:`G^{(n)}_t=R^{(n)}_t+\\gamma^nQ(S_{t+n}, A_{t+n})`.

        Parameters
        ----------
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

                G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n},A_{t+n})


        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 2d Tensor, dtype: int, shape: [batch_size, num_actions]

            A batch of (potential) next actions :term:`A_next`. This argument
            is only used if ``update_strategy='sarsa'``.

        Returns
        -------
        Gn : 1d array, dtype: int, shape: [batch_size]

            A batch of bootstrap-estimated returns
            :math:`G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n},A_{t+n})` computed
            according to given ``update_strategy``.

        """
        if self.update_strategy == 'sarsa':
            assert A_next is not None
            Q_next = self.batch_eval(
                S_next, use_target_model=self.bootstrap_with_target_model)
            Q_next = np.einsum('ij,ij->i', Q_next, A_next)

        elif self.update_strategy == 'q_learning':
            Q_next = self.batch_eval(
                S_next, use_target_model=self.bootstrap_with_target_model)
            Q_next = np.max(Q_next, axis=1)  # greedy

        elif self.update_strategy == 'double_q_learning':
            if not self.bootstrap_with_target_model:
                raise ValueError(
                    "incompatible settings: "
                    "update_strategy='double_q_learning' requires that "
                    "bootstrap_with_target_model=True")
            A_next = np.argmax(
                self.batch_eval(S_next, use_target_model=False), axis=1)
            Q_next = self.batch_eval(S_next, use_target_model=True)
            Q_next = project_onto_actions_np(Q_next, A_next)

        else:
            raise ValueError("unknown update_strategy")

        Gn = Rn + In * Q_next
        return Gn

    @abstractmethod
    def batch_eval(self, S, A=None, use_target_model=False):
        """
        Evaluate the Q-function on a batch of state (or state-action)
        observations.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : 1d array, dtype: int, shape: [batch_size], optional

            A batch of actions that were taken.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        Q : 1d or 2d array of floats

            If action ``A`` is provided, a 1d array representing a batch of
            :math:`q(s,a)` is returned. If, on the other hand, ``A`` is left
            unspecified, a vector representing a batch of :math:`q(s,.)` is
            returned instead. The shape of the latter return value is
            ``[batch_size, num_actions]``, which is only well-defined for
            discrete action
            spaces.

        """
        pass


class QTypeI(BaseQ):
    """
    A :term:`type-I state-action value function` :math:`(s,a)\\mapsto q(s,a)`.

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

    update_strategy : str, optional

        The update strategy that we use to select the (would-be) next-action
        :math:`A_{t+n}` in the bootstrapped target:

        .. math::

            G^{(n)}_t\\ =\\ R^{(n)}_t + \\gamma^n Q(S_{t+n}, A_{t+n})

        Options are:

            'sarsa'
                Sample the next action, i.e. use the action that was actually
                taken.

            'q_learning'
                Take the action with highest Q-value under the current
                estimate, i.e. :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n}, a)`.
                This is an off-policy method.

            'double_q_learning'
                Same as 'q_learning', :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n},
                a)`, except that the value itself is computed using the
                :term:`target_model` rather than the primary model, i.e.

                .. math::

                    A_{t+n}\\ &=\\
                        \\arg\\max_aQ_\\text{primary}(S_{t+n}, a)\\\\
                    G^{(n)}_t\\ &=\\ R^{(n)}_t
                        + \\gamma^n Q_\\text{target}(S_{t+n}, A_{t+n})

            'expected_sarsa'
                Similar to SARSA in that it's on-policy, except that we take
                the expectated Q-value rather than a sample of it, i.e.

                .. math::

                    G^{(n)}_t\\ =\\ R^{(n)}_t
                        + \\gamma^n\\sum_a\\pi(a|s)\\,Q(S_{t+n}, a)

    """
    def batch_eval(self, S, A=None, use_target_model=False):
        model = self.target_model if use_target_model else self.predict_model

        if A is not None:
            Q = model.predict_on_batch([S, A])
            check_numpy_array(Q, ndim=2, axis_size=1, axis=1)
            Q = np.squeeze(Q, axis=1)
            return Q  # shape: [batch_size]
        else:
            Q = []
            for a in range(self.num_actions):
                A = one_hot(a * np.ones(len(S), dtype='int'), self.num_actions)
                Q.append(self.batch_eval(S, A))
            Q = np.stack(Q, axis=1)
            check_numpy_array(Q, ndim=2, axis_size=self.num_actions, axis=1)
            return Q  # shape: [batch_size, num_actions]

    def batch_update(self, S, A, Rn, In, S_next, A_next=None):
        G = self.bootstrap_target(Rn, In, S_next, A_next)
        losses = self._train_on_batch([S, A], G)
        return losses

    def _init_models(self):

        # extract input shapes
        s_shape = self.env.observation_space.shape
        s_dtype = self.env.observation_space.dtype
        if self.action_space_is_discrete:
            a_shape = [self.num_actions]
            a_dtype = 'float'
        else:
            a_shape = self.env.action_space.shape
            a_dtype = self.env.action_space.dtype

        # input
        S = keras.Input(name='value/S', shape=s_shape, dtype=s_dtype)
        A = keras.Input(name='value/A', shape=a_shape, dtype=a_dtype)

        # forward pass
        X = self.function_approximator.body_q1(S, A)
        Q = self.function_approximator.head_q1(X)

        # regular models
        self.train_model = keras.Model([S, A], Q)
        self.train_model.compile(
            loss=self.function_approximator.VALUE_LOSS_FUNCTION,
            optimizer=self.function_approximator.optimizer)

        # predict and target model
        self.predict_model = self.train_model  # yes, it's trivial for type-I
        self.target_model = keras.models.clone_model(self.predict_model)


class QTypeII(BaseQ):
    """
    A :term:`type-II state-action value function` :math:`s\\mapsto q(s,.)`.

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

    update_strategy : str, optional

        The update strategy that we use to select the (would-be) next-action
        :math:`A_{t+n}` in the bootstrapped target:

        .. math::

            G^{(n)}_t\\ =\\ R^{(n)}_t + \\gamma^n Q(S_{t+n}, A_{t+n})

        Options are:

            'sarsa'
                Sample the next action, i.e. use the action that was actually
                taken.

            'q_learning'
                Take the action with highest Q-value under the current
                estimate, i.e. :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n}, a)`.
                This is an off-policy method.

            'double_q_learning'
                Same as 'q_learning', :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n},
                a)`, except that the value itself is computed using the
                :term:`target_model` rather than the primary model, i.e.

                .. math::

                    A_{t+n}\\ &=\\
                        \\arg\\max_aQ_\\text{primary}(S_{t+n}, a)\\\\
                    G^{(n)}_t\\ &=\\ R^{(n)}_t
                        + \\gamma^n Q_\\text{target}(S_{t+n}, A_{t+n})

            'expected_sarsa'
                Similar to SARSA in that it's on-policy, except that we take
                the expectated Q-value rather than a sample of it, i.e.

                .. math::

                    G^{(n)}_t\\ =\\ R^{(n)}_t
                        + \\gamma^n\\sum_a\\pi(a|s)\\,Q(S_{t+n}, a)

    """
    def batch_eval(self, S, A=None, use_target_model=False):
        model = self.target_model if use_target_model else self.predict_model

        if A is not None:
            Q = model.predict_on_batch(S)  # shape: [batch_size, num_actions]
            check_numpy_array(Q, ndim=2, axis_size=self.num_actions, axis=1)
            check_numpy_array(
                A, ndim=1, dtype='int', axis_size=Q.shape[0], axis=0)
            Q = project_onto_actions_np(Q, A)
            return Q  # shape: [batch_size]
        else:
            Q = model.predict_on_batch(S)
            check_numpy_array(Q, ndim=2, axis_size=self.num_actions, axis=1)
            return Q  # shape: [batch_size, num_actions]

    def batch_update(self, S, A, Rn, In, S_next, A_next=None):
        G = self.bootstrap_target(Rn, In, S_next, A_next)
        losses = self._train_on_batch([S, G], A)
        return losses

    def _init_models(self):
        if not self.action_space_is_discrete:
            raise ActionSpaceError(
                "QTypeII is incompatible with non-discrete action spaces; "
                "please use QTypeI instead")

        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)
        G = keras.Input(name='value/G', shape=(), dtype='float')

        # forward pass
        X = self.function_approximator.body(S)
        Q = self.function_approximator.head_q2(X)

        # loss
        loss = ProjectedSemiGradientLoss(
            G, base_loss=self.function_approximator.VALUE_LOSS_FUNCTION)

        # regular models
        self.train_model = keras.Model([S, G], Q)
        self.train_model.compile(
            loss=loss, optimizer=self.function_approximator.optimizer)

        # predict and target model
        self.predict_model = keras.Model(S, Q)
        self.target_model = keras.models.clone_model(self.predict_model)
