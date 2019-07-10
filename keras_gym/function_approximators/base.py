from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..base.errors import MissingModelError
from ..base.mixins import RandomStateMixin, NumActionsMixin, LoggerMixin
from ..policies.base import BasePolicy
from ..caching import NStepCache
from ..losses import SoftmaxPolicyLossWithLogits, ClippedSurrogateLoss
from ..utils import (
    project_onto_actions_np, softmax, argmax, check_numpy_array)


__all__ = (
    'BaseV',
    'BaseQTypeI',
    'BaseQTypeII',
    'BaseSoftmaxPolicy',
)


class BaseFunctionApproximator(ABC, LoggerMixin):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_eval(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_update(self, *args, **kwargs):
        pass

    def _check_attrs(self):
        required_attrs = [
            'env', 'gamma', 'bootstrap_n', 'train_model', 'predict_model',
            'target_model', '_cache']

        if isinstance(self, BaseSoftmaxPolicy):
            required_attrs.remove('bootstrap_n')
            required_attrs.remove('gamma')
            required_attrs.remove('_cache')

        missing_attrs = ", ".join(
            attr for attr in required_attrs if not hasattr(self, attr))

        if missing_attrs:
            raise AttributeError(
                "missing attributes: {}".format(missing_attrs))

    def _train_on_batch(self, inputs, outputs):
        """
        Run self.train_model.train_on_batch(inputs, outputs) and return the
        losses as a dict of type: {loss_name <str>: loss_value <float>}.

        """
        losses = self.train_model.train_on_batch(inputs, outputs)

        # add metric names
        if len(self.train_model.metrics_names) > 1:
            assert len(self.train_model.metrics_names) == len(losses)
            losses = dict(zip(self.train_model.metrics_names, losses))
        else:
            assert isinstance(losses, (float, np.float32, np.float64))
            assert len(self.train_model.metrics_names) == 1
            losses = {self.train_model.metrics_names[0]: losses}

        if hasattr(self.env, 'record_losses'):
            self.env.record_losses(losses)

        return losses

    def sync_target_model(self, tau=1.0):
        """
        Synchronize the target model with the primary model.

        Parameters
        ----------
        tau : float between 0 and 1, optional

            The amount of exponential smoothing to apply in the target update:

            .. math::

                w_\\text{target}\\ \\leftarrow\\ (1 - \\tau)\\,w_\\text{target}
                + \\tau\\,w_\\text{primary}

        """
        if tf.__version__ >= '2.0':
            target_weights = self.target_model.trainable_variables
            primary_weights = self.predict_model.trainable_variables
            tf.group(*(
                K.update(wt, wt + tau * (wp - wt))
                for wt, wp in zip(target_weights, primary_weights)))
            self.logger.debug(
                "updated target_mode with tau = {:.3g}".format(tau))

        else:
            if not hasattr(self, '_target_model_sync_op'):
                target_weights = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='target')  # list
                primary_weights = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='primary')  # list

                if not target_weights:
                    raise MissingModelError(
                        "no model weights found in variable scope: 'target'")
                if not primary_weights:
                    raise MissingModelError(
                        "no model weights found in variable scope: 'primary'")
                assert len(primary_weights) == len(target_weights)

                self._target_model_sync_tau = tf.placeholder(
                    tf.float32, shape=())
                self._target_model_sync_op = tf.group(*(
                    K.update(wt, wt + self._target_model_sync_tau * (wp - wt))
                    for wt, wp in zip(target_weights, primary_weights)))

            K.get_session().run(
                self._target_model_sync_op,
                feed_dict={self._target_model_sync_tau: tau})
            self.logger.debug(
                "updated target_mode with tau = {:.3g}".format(tau))


class BaseV(BaseFunctionApproximator):
    """
    Base class for modeling a :term:`state value function`.

    A :term:`state value function` is implemented by mapping :math:`s\\mapsto
    v(s)`.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model(:term:`S`, :term:`V`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`V`)

        Used for predicting. For a :term:`state value function` the
        :term:`target_model` and :term:`predict_model` are the same.

    target_model : keras.Model(:term:`S`, :term:`V`)

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

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
            self, env, train_model, predict_model, target_model,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)
        self.bootstrap_with_target_model = bool(bootstrap_with_target_model)

        self._cache = NStepCache(self.env, self.bootstrap_n, self.gamma)

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


class BaseGenericQ(BaseFunctionApproximator, NumActionsMixin):
    UPDATE_STRATEGIES = ('sarsa', 'q_learning', 'double_q_learning')

    def __init__(
            self, env, train_model, predict_model, target_model,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='sarsa'):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)
        self.bootstrap_with_target_model = bool(bootstrap_with_target_model)
        self.update_strategy = update_strategy

        self._cache = NStepCache(self.env, self.bootstrap_n, self.gamma)

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
            A = np.expand_dims(a, axis=0)
            Q = self.batch_eval(S, A, use_target_model=use_target_model)
            check_numpy_array(Q, shape=(1,))
            Q = np.asscalar(Q)
        else:
            Q = self.batch_eval(S, use_target_model=use_target_model)
            check_numpy_array(Q, shape=(1, self.num_actions))
            Q = np.squeeze(Q, axis=0)
        return Q

    def update(self, s, pi, r, done):
        """
        Update the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        pi : int or 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. This may
            be just an indicator if the action propensities are inferred
            through sampling. For instance, let's say our action space is
            :class:`Discrete(4)`, then passing ``pi = 2`` is equivalent to
            passing ``pi = [0, 0, 1, 0]``. Both would indicate that the action
            :math:`a=2` was drawn from the behavior policy.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        assert self.env.observation_space.contains(s)
        pi = self.check_pi(pi)
        self._cache.add(s, pi, r, done)

        # eager updates
        while self._cache:
            self.batch_update(*self._cache.pop())  # pop with batch_size=1

    def batch_update(self, S, P, Rn, In, S_next, P_next=None):
        """
        Update the value function on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        P : 2d Tensor, dtype: int, shape: [batch_size]

            A batch of action propensities. :term:`P` is typically just an
            indicator for which action was chosen by the behavior policy. In
            this sense, :term:`P` acts as a projector more than a prediction
            target. That is, :term:`P` is used to project our predicted values
            down to those for which we actually received the feedback signal.

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

        P_next : 2d Tensor, dtype: int, shape: [batch_size, num_actions]

            Action propensities :term:`P_next` for the (potential) next action.
            This argument is only used if ``update_strategy='sarsa'``.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        G = self.bootstrap_target(Rn, In, S_next, P_next)
        losses = self._train_on_batch([S, G], P)
        return losses

    def bootstrap_target(self, Rn, In, S_next, P_next=None):
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

        P_next : 2d Tensor, dtype: int, shape: [batch_size, num_actions]

            Action propensities :term:`P_next` for the (potential) next action.
            This argument is only used if ``update_strategy='sarsa'``.

        Returns
        -------
        Gn : 1d array, dtype: int, shape: [batch_size]

            A batch of bootstrap-estimated returns
            :math:`G^{(n)}_t=R^{(n)}_t+I^{(n)}_tQ(S_{t+n},A_{t+n})` computed
            according to given ``update_strategy``.

        """
        if self.update_strategy == 'sarsa':
            assert P_next is not None
            Q_next = self.batch_eval(
                S_next, use_target_model=self.bootstrap_with_target_model)
            Q_next = np.einsum('ij,ij->i', Q_next, P_next)
        elif self.update_strategy == 'q_learning':
            Q_next = np.max(
                self.batch_eval(
                    S_next, use_target_model=self.bootstrap_with_target_model),
                axis=1)
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


class BaseQTypeI(BaseGenericQ):
    """
    Base class for modeling :term:`type-I <type-I state-action value function>`
    Q-function.

    A :term:`type-I <type-I state-action value function>` Q-function is
    implemented by mapping :math:`(s, a)\\mapsto q(s,a)`.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model([:term:`S`, :term:`A`], :term:`Q_sa`)

        Used for training.

    predict_model : keras.Model([:term:`S`, :term:`A`], :term:`Q_sa`)

        Used for predicting. For :term:`type-I <type-I state-action value
        function>` Q-functions, the :term:`target_model` and
        :term:`predict_model` are the same.

    target_model : keras.Model(:term:`S`, :term:`Q_sa`)

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

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
                A = a * np.ones(len(S), dtype='int')
                Q.append(self.batch_eval(S, A))
            Q = np.stack(Q, axis=1)
            check_numpy_array(Q, ndim=2, axis_size=self.num_actions, axis=1)
            return Q  # shape: [batch_size, num_actions]


class BaseQTypeII(BaseGenericQ):
    """
    Base class for modeling :term:`type-II <type-II state-action value
    function>` Q-function.

    A :term:`type-II <type-II state-action value function>` Q-function is
    implemented by mapping :math:`s\\mapsto q(s,.)`.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model([:term:`S`, :term:`G`], :term:`Q_s`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`Q_s`)

        Used for predicting.

    target_model : keras.Model(:term:`S`, :term:`Q_s`)

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

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
        :math:`A_{t+n}` in the bootsrapped target:

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


class BaseSoftmaxPolicy(BasePolicy, BaseFunctionApproximator, NumActionsMixin, RandomStateMixin):  # noqa: E501
    """
    Base class for modeling :term:`updateable policies <updateable policy>` for
    discrete action spaces.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model([:term:`S`, :term:`Adv`], :term:`Z`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`Z`)

        Used for predicting.

    target_model : keras.Model(:term:`S`, :term:`Z`)

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

    update_strategy : str, optional

        The strategy for updating our policy. This typically determines the
        loss function that we use for our policy function approximator.

        Options are:

            'vanilla'
                Plain vanilla policy gradient. The corresponding (surrogate)
                loss function that we use is:

                .. math::

                    J(\\theta)\\ =\\ -\\mathcal{A}(s,a)\\,\\ln\\pi(a|s,\\theta)

            'ppo'
                `Proximal policy optimization
                <https://arxiv.org/abs/1707.06347>`_ uses a clipped proximal
                loss:

                .. math::

                    J(\\theta)\\ =\\ \\min\\Big(
                        r(\\theta)\\,\\mathcal{A}(s,a)\\,,\\
                        \\text{clip}\\big(
                            r(\\theta), 1-\\epsilon, 1+\\epsilon\\big)
                                \\,\\mathcal{A}(s,a)\\Big)

                where :math:`r(\\theta)` is the probability ratio:

                .. math::

                    r(\\theta)\\ =\\ \\frac
                        {\\pi(a|s,\\theta)}
                        {\\pi(a|s,\\theta_\\text{old})}

            'cross_entropy'
                Straightforward categorical cross-entropy (from logits). This
                loss function does *not* make use of the advantages
                :term:`Adv`. Instead, it minimizes the cross entropy between
                the behavior policy :math:`\\pi_b(a|s)` and the learned policy
                :math:`\\pi_\\theta(a|s)`:

                .. math::

                    J(\\theta)\\ =\\ \\hat{\\mathbb{E}}_t\\left\\{
                        -\\sum_a \\pi_b(a|S_t)\\, \\log \\pi_\\theta(a|S_t)
                    \\right\\}

    ppo_clipping : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    UPDATE_STRATEGIES = ('vanilla', 'ppo', 'cross_entropy')

    def __init__(
            self, env, train_model, predict_model, target_model,
            update_strategy='vanilla',
            ppo_clipping=0.2,
            entropy_bonus=0.01,
            random_seed=None):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.update_strategy = update_strategy
        self.ppo_clipping = float(ppo_clipping)
        self.entropy_bonus = float(entropy_bonus)
        self.random_seed = random_seed  # sets self.random in RandomStateMixin

        # TODO: allow for non-discrete action spaces
        self._actions = np.arange(self.num_actions)

    def __call__(self, s, use_target_model=False):
        """
        Draw an action from the current policy :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        a : action

            A single action proposed under the current policy.

        """
        proba = self.proba(s, use_target_model=use_target_model)
        a = self.random.choice(self._actions, p=proba)
        return a

    def proba(self, s, use_target_model=False):
        """
        Get the probabilities over all actions :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        pi : 1d array, shape: [num_actions]

            Probabilities over all actions.

            **Note.** This hasn't yet been implemented for non-discrete action
            spaces.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        P = self.batch_eval(S, use_target_model=use_target_model)
        check_numpy_array(P, shape=(1, self.num_actions))
        pi = np.squeeze(P, axis=0)
        return pi

    def greedy(self, s, use_target_model=False):
        """
        Draw the greedy action, i.e. :math:`\\arg\\max_a\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        a : action

            A single action proposed under the current policy.

        """
        a = argmax(self.proba(s, use_target_model=use_target_model))
        return a

    def update(self, s, pi, advantage):
        """
        Update the policy.

        Parameters
        ----------
        s : state observation

            A single state observation.

        pi : 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. This may
            be just an indicator if the action propensities are inferred
            through sampling, e.g. if we have four possible actions ``pi = [0,
            0, 1, 0]`` would indicate that the action :math:`a=2` was drawn
            from the behavior policy.

        advantage : float

            A value for the advantage :math:`\\mathcal{A}(s,a) = q(s,a) -
            v(s)`. This might be sampled and/or estimated version of the true
            advantage.

        """
        assert self.env.observation_space.contains(s)
        check_numpy_array(pi, ndim=1, axis_size=self.num_actions, axis=0)

        S = np.expand_dims(s, axis=0)
        P = np.expand_dims(pi, axis=0)
        Adv = np.expand_dims(advantage, axis=0)
        self.batch_update(S, P, Adv)

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
        P : 2d array, shape: [batch_size, num_actions]

            A batch of predicted action probabilities :math:`\\pi(a|s)`.

        """
        model = self.target_model if use_target_model else self.predict_model

        Z = model.predict_on_batch(S)
        check_numpy_array(Z, ndim=2, axis_size=self.num_actions, axis=1)
        P = softmax(Z, axis=1)
        return P  # shape: [batch_size, num_actions]

    def batch_update(self, S, P, Adv):
        """
        Update the policy on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        P : 2d Tensor, dtype: int, shape: [batch_size]

            A batch of action propensities. :term:`P` is typically just an
            indicator for which action was chosen by the behavior policy. In
            this sense, :term:`P` acts as a projector more than a prediction
            target. That is, :term:`P` is used to project our predicted values
            down to those for which we actually received the feedback signal:
            :term:`Adv`.

        Adv : 1d array, dtype: float, shape: [batch_size]

            A value for the :term:`advantage <Adv>` :math:`\\mathcal{A}(s,a) = q(s,a) -
            v(s)`. This might be sampled and/or estimated version of the true
            advantage.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        check_numpy_array(P, ndim=2, axis_size=self.num_actions, axis=1)
        losses = self._train_on_batch([S, Adv], P)
        return losses

    def _policy_loss(self, Adv, Z_target=None):
        if self.update_strategy == 'vanilla':
            return SoftmaxPolicyLossWithLogits(
                Adv, entropy_bonus=self.entropy_bonus)

        if self.update_strategy == 'ppo':
            assert Z_target is not None
            return ClippedSurrogateLoss(
                Adv, Z_target, entropy_bonus=self.entropy_bonus,
                epsilon=self.ppo_clipping)

        if self.update_strategy == 'cross_entropy':
            return keras.losses.CategoricalCrossentropy(from_logits=True)

        raise ValueError(
            "unknown update_strategy '{}'".format(self.update_strategy))
