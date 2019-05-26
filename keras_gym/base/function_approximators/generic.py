from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ...utils import (
    project_onto_actions_np, project_onto_actions_tf, check_numpy_array,
    softmax, argmax)
from ...caching import NStepCache
from ..errors import MissingModelError
from ..mixins import RandomStateMixin, NumActionsMixin, LoggerMixin
from ..policy import BasePolicy


__all__ = (
    'GenericV',
    'GenericQTypeI',
    'GenericQTypeII',
    'GenericSoftmaxPolicy',
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
            'target_model', 'bootstrap_model', '_cache']

        if isinstance(self, GenericSoftmaxPolicy):
            required_attrs.remove('bootstrap_model')
            required_attrs.remove('bootstrap_n')
            required_attrs.remove('gamma')
            required_attrs.remove('_cache')

        missing_attrs = ", ".join(
            attr for attr in required_attrs if not hasattr(self, attr))

        if missing_attrs:
            raise AttributeError(
                "missing attributes: {}".format(missing_attrs))

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

            assert len(primary_weights) == len(target_weights), "incompatible"

            self._target_model_sync_tau = tf.placeholder(tf.float32, shape=())
            self._target_model_sync_op = tf.group(*(
                K.update(wt, wt + self._target_model_sync_tau * (wp - wt))
                for wt, wp in zip(target_weights, primary_weights)))

        K.get_session().run(
            self._target_model_sync_op,
            feed_dict={self._target_model_sync_tau: tau})
        self.logger.info("updated target_model")


class GenericV(BaseFunctionApproximator):
    """
    Base class for modeling a :term:`state value function`.

    A :term:`state value function` is implemented by mapping :math:`s\\mapsto
    V(s)`.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model(:term:`S`, :term:`V`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`V`)

        Used for predicting. For a :term:`state value function` the
        :term:`target_model` and :term:`predict_model` are the same.

    target_model : keras.Model(:term:`S`, :term:`V`), optional

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

    bootstrap_model : keras.Model([:term:`S`, :term:`Rn`, :term:`I_next`, :term:`S_next`]), optional

        A :term:`bootstrap_model` can be used for training. It differs from the
        :term:`train_model` in that it computes the bootstrapped target
        internally (instead of receiving it as input). The use of a
        :term:`bootstrap_model` can speed up the training process. Notice that
        the :term:`bootstrap_model` requires no external output. Instead, the
        output is passed to the model using keras's ``target_tensors`` kwarg.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

    """  # noqa: E501
    def __init__(
            self, env, train_model, predict_model,
            target_model=None,
            bootstrap_model=None,
            gamma=0.9,
            bootstrap_n=1):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.bootstrap_model = bootstrap_model
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)

        self._cache = NStepCache(self.bootstrap_n, self.gamma)

    def __call__(self, s):
        """
        Evaluate the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        V : float or array of floats

            The estimated value of the state :math:`V(s)`.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        V = self.batch_eval(S)
        check_numpy_array(V, shape=(1,))
        V = np.squeeze(V, axis=0)
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
            S, _, Rn, I_next, S_next, _ = self._cache.pop()
            self.batch_update(S, Rn, I_next, S_next)

    def batch_update(self, S, Rn, I_next, S_next):
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

        I_next : 1d array, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I_t=\\gamma^n` if the episode is ongoing
            and :math:`I_t=0` otherwise. This allows us to write the
            bootstrapped target as :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},
            A_{t+n})`.

        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        """
        if self.bootstrap_model is not None:
            self.bootstrap_model.train_on_batch([S, Rn, I_next, S_next])
        else:
            V_next = self.batch_eval(S_next, use_target_model=True)
            Gn = Rn + I_next * V_next
            self.train_model.train_on_batch(S, Gn)

    def batch_eval(self, S, use_target_model=False):
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.predict_model

        V = model.predict_on_batch(S)
        check_numpy_array(V, ndim=2, axis_size=1, axis=1)
        V = np.squeeze(V, axis=1)  # shape: [batch_size]
        return V


class BaseGenericQ(BaseFunctionApproximator, NumActionsMixin):
    UPDATE_STRATEGIES = ('sarsa', 'q_learning', 'double_q_learning')

    def __init__(
            self, env, train_model, predict_model,
            target_model=None,
            bootstrap_model=None,
            gamma=0.9,
            bootstrap_n=1,
            update_strategy='sarsa'):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.bootstrap_model = bootstrap_model
        self.gamma = float(gamma)
        self.bootstrap_n = int(bootstrap_n)
        self.update_strategy = update_strategy

        self._cache = NStepCache(self.bootstrap_n, self.gamma)

    def __call__(self, s, a=None):
        """
        Evaluate the Q-function.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action, optional

            A single action.

        Returns
        -------
        Q : float or array of floats

            If action ``a`` is provided, a single float representing
            :math:`Q(s,a)` is returned. If, on the other hand, ``a`` is left
            unspecified, a vector representing :math:`Q(s,.)` is returned
            instead. The shape of the latter return value is ``[num_actions]``,
            which is only well-defined for discrete action spaces.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        if a is not None:
            assert self.env.action_space.contains(a)
            A = np.expand_dims(a, axis=0)
            Q = self.batch_eval(S, A)
            check_numpy_array(Q, shape=(1,))
            Q = np.squeeze(Q, axis=0)
        else:
            Q = self.batch_eval(S)
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
        Update the value function on a batch of transitions.

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

        A_next : 1d array, dtype: int, shape: [batch_size]

            A batch of next-actions that were taken. This is only required for
            SARSA (on-policy) updates.

        """
        if self.bootstrap_model is not None:
            self.bootstrap_model.train_on_batch(
                [S, Rn, I_next, S_next, A_next], A)
        else:
            G = self.bootstrap_target_np(Rn, I_next, S_next, A_next)
            self.train_model.train_on_batch([S, G], A)

    def bootstrap_target_np(self, Rn, I_next, S_next, A_next=None):
        """
        Get the bootstrapped target
        :math:`G^{(n)}_t=R^{(n)}_t+\\gamma^nQ(S_{t+n}, A_{t+n})`.

        This is the *numpy* implementation.

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

        I_next : 1d array, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I_t=\\gamma^n` if the episode is ongoing
            and :I_t=0: otherwise. This allows us to write the bootstrapped
            target as :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},A_{t+n})`.

        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 1d array, dtype: int, shape: [batch_size], optional

            A batch of next-actions that were taken.

        Returns
        -------
        Gn : 1d array, dtype: int, shape: [batch_size]

            A batch of bootstrap-estimated returns
            :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},A_{t+n})` computed
            according to given ``update_strategy``.

        """
        if self.update_strategy == 'sarsa':
            assert A_next is not None
            Q_next = self.batch_eval(S_next, A_next, use_target_model=True)
        elif self.update_strategy == 'q_learning':
            Q_next = np.max(
                self.batch_eval(S_next, use_target_model=True), axis=1)
        elif self.update_strategy == 'double_q_learning':
            A_next = np.argmax(
                self.batch_eval(S_next, use_target_model=False), axis=1)
            Q_next = self.batch_eval(S_next, use_target_model=True)
            Q_next = project_onto_actions_np(Q_next, A_next)
        else:
            raise ValueError("unknown update_strategy")

        Gn = Rn + I_next * Q_next
        return Gn

    def bootstrap_target_tf(self, Rn, I_next, S_next, A_next=None):
        """
        Get the bootstrapped target
        :math:`G^{(n)}_t=R^{(n)}_t+\\gamma^nQ(S_{t+n}, A_{t+n})`.

        This is the *tensorflow* implementation.

        Parameters
        ----------
        Rn : 1d tensor, dtype: float, shape: [batch_size]

            A batch of partial returns. For example, in n-step bootstrapping
            this is given by:

            .. math::

                R^{(n)}_t\\ =\\ R_t + \\gamma\\,R_{t+1} + \\dots
                    \\gamma^{n-1}\\,R_{t+n-1}

            In other words, it's the non-bootstrapped part of the n-step
            return.

        I_next : 1d tensor, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I_t=\\gamma^n` if the episode is ongoing
            and :I_t=0: otherwise. This allows us to write the bootstrapped
            target as :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},A_{t+n})`.

        S_next : nd tensor, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 1d tensor, dtype: int, shape: [batch_size], optional

            A batch of next-actions that were taken.

        Returns
        -------
        Gn : 1d tensor, dtype: int, shape: [batch_size]

            A batch of bootstrap-estimated returns
            :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},A_{t+n})` computed
            according to given ``update_strategy``.

        """
        raise NotImplementedError('bootstrap_target_tf')

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

            Whether to use the ``target_model``. If ``False`` (default), the
            primary ``predict_model`` is used.

        Returns
        -------
        Q : 1d or 2d array of floats

            If action ``A`` is provided, a 1d array representing a batch of
            :math:`Q(s,a)` is returned. If, on the other hand, ``A`` is left
            unspecified, a vector representing a batch of :math:`Q(s,.)` is
            returned instead. The shape of the latter return value is
            ``[batch_size, num_actions]``, which is only well-defined for
            discrete action
            spaces.

        """
        pass


class GenericQTypeI(BaseGenericQ):
    """
    Base class for modeling :term:`type-I <type-I state-action value function>`
    Q-function.

    A :term:`type-I <type-I state-action value function>` Q-function is
    implemented by mapping :math:`(s, a)\\mapsto Q(s,a)`.

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

    target_model : keras.Model(:term:`S`, :term:`Q_sa`), optional

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

    bootstrap_model : keras.Model([:term:`S`, :term:`A`, :term:`Rn`, :term:`I_next`, :term:`S_next`, :term:`A_next`]), optional

        A :term:`bootstrap_model` can be used for training. It differs from the
        :term:`train_model` in that it computes the bootstrapped target
        internally (instead of receiving it as input). The use of a
        :term:`bootstrap_model` can speed up the training process. Notice that
        the :term:`bootstrap_model` requires no external output. Instead, the
        output is passed to the model using keras's ``target_tensors`` kwarg.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

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

            'expected_sarsa'
                Similar to SARSA in that it's on-policy, except that we take
                the expectated Q-value rather than a sample of it, i.e.

                .. math::

                    G^{(n)}_t\\ =\\ R^{(n)}_t
                        + \\gamma^n\\sum_a\\pi(a|s)\\,Q(S_{t+n}, a)

    """  # noqa: E501
    def batch_eval(self, S, A=None, use_target_model=False):
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.predict_model

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


class GenericQTypeII(BaseGenericQ):
    """
    Base class for modeling :term:`type-II <type-II state-action value
    function>` Q-function.

    A :term:`type-II <type-II state-action value function>` Q-function is
    implemented by mapping :math:`s\\mapsto Q(s,.)`.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model([:term:`S`, :term:`G`], :term:`Q_s`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`Q_s`)

        Used for predicting.

    target_model : keras.Model(:term:`S`, :term:`Q_s`), optional

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

    bootstrap_model : keras.Model([:term:`S`, :term:`Rn`, :term:`I_next`, :term:`S_next`, :term:`A_next`], :term:`Q_sa`), optional

        A :term:`bootstrap_model` can be used for training. It differs from the
        :term:`train_model` in that it computes the bootstrapped target
        internally (instead of receiving it as input). The use of a
        :term:`bootstrap_model` can speed up the training process.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

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

            'expected_sarsa'
                Similar to SARSA in that it's on-policy, except that we take
                the expectated Q-value rather than a sample of it, i.e.

                .. math::

                    G^{(n)}_t\\ =\\ R^{(n)}_t
                        + \\gamma^n\\sum_a\\pi(a|s)\\,Q(S_{t+n}, a)

    """  # noqa: E501
    def batch_eval(self, S, A=None, use_target_model=False):
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.predict_model

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

    def bootstrap_target_tf(self, Rn, I_next, S_next, A_next=None):
        if self.target_model is None:
            raise MissingModelError('target_model')

        if self.update_strategy == 'sarsa':
            assert A_next is not None
            Q_next = self.target_model(S_next)
            Q_next = project_onto_actions_tf(Q_next, A_next)
        elif self.update_strategy == 'q_learning':
            Q_next = self.target_model(S_next)
            Q_next = K.max(Q_next, axis=1)
        elif self.update_strategy == 'double_q_learning':
            A_next = K.argmax(self.predict_model(S_next), axis=1)
            Q_next = self.target_model(S_next)
            Q_next = project_onto_actions_np(Q_next, A_next)
        else:
            raise ValueError(
                "unknown update_strategy: {}".format(self.update_strategy))

        Gn = Rn + I_next * Q_next
        return Gn


class GenericSoftmaxPolicy(
        BasePolicy, BaseFunctionApproximator, NumActionsMixin,
        RandomStateMixin):

    """
    Base class for modeling :term:`updateable policies <updateable policy>` for
    discrete action spaces.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    train_model : keras.Model([:term:`S`, :term:`Adv`], :term:`Logits`)

        Used for training.

    predict_model : keras.Model(:term:`S`, :term:`Logits`)

        Used for predicting.

    target_model : keras.Model(:term:`S`, :term:`Logits`), optional

        A :term:`target_model` is used to make predictions on a bootstrapping
        scenario. It can be advantageous to use a point-in-time copy of the
        :term:`predict_model` to construct a bootstrapped target.

    bootstrap_model : keras.Model([:term:`S`, :term:`Rn`, :term:`I_next`, :term:`S_next`, :term:`A_next`], :term:`Logits`), optional

        A :term:`bootstrap_model` can be used for training. It differs from the
        :term:`train_model` in that it computes the bootstrapped target
        internally (instead of receiving it as input). The use of a
        :term:`bootstrap_model` can speed up the training process.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

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

                #TODO: to be implemented -Kris


    """  # noqa: E501

    UPDATE_STRATEGIES = ('vanilla', 'trpo', 'ppo')

    def __init__(
            self, env, train_model, predict_model,
            target_model=None,
            update_strategy='vanilla',
            random_seed=None):

        self.env = env
        self.train_model = train_model
        self.predict_model = predict_model
        self.target_model = target_model
        self.update_strategy = update_strategy
        self.random_seed = random_seed  # sets self.random in RandomStateMixin

        # TODO: allow for non-discrete action spaces
        self._actions = np.arange(self.num_actions)

    def __call__(self, s):
        a = self.random.choice(self._actions, p=self.proba(s))
        return a

    def proba(self, s):
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        Pi = self.batch_eval(S)
        check_numpy_array(Pi, shape=(1, self.num_actions))
        pi = np.squeeze(Pi, axis=0)
        return pi

    def greedy(self, s):
        a = argmax(self.proba(s))
        return a

    def update(self, s, a, advantage):
        """
        Update the policy.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action that was taken.

        advantage : float

            A value for the advantage :math:`\\mathcal{A}(s,a)=Q(s,a)-V(s)`.
            This might be sampled and/or estimated version of the true
            advantage.

        """
        assert self.env.observation_space.contains(s)
        assert self.env.action_space.contains(a)
        S = np.expand_dims(s, axis=0)
        A = np.expand_dims(a, axis=0)
        Adv = np.expand_dims(advantage, axis=0)
        self.batch_update(S, A, Adv)

    def batch_eval(self, S, use_target_model=False):
        """
        Evaluate the policy on a batch of state observations.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        use_target_model : bool, optional

            Whether to use the ``target_model``. If ``False`` (default), the
            primary ``predict_model`` is used.

        """
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.predict_model

        Logits = model.predict_on_batch(S)
        check_numpy_array(Logits, ndim=2, axis_size=self.num_actions, axis=1)
        Pi = softmax(Logits, axis=1)
        return Pi  # shape: [batch_size, num_actions]

    def batch_update(self, S, A, Adv):
        """
        Update the policy on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : 1d array, dtype: int, shape: [batch_size]

            A batch of actions that were taken.

        Adv : 1d array, dtype: float, shape: [batch_size]

            A value for the advantage :math:`\\mathcal{A}(s,a)=Q(s,a)-V(s)`.
            This might be sampled and/or estimated version of the true
            advantage.

        """
        self.train_model.train_on_batch([S, Adv], A)
