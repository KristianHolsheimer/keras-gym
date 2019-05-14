from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ...utils import (
    project_onto_actions_np, check_numpy_array, softmax, argmax)
from ...caching import NStepCache
from ..errors import MissingModelError
from ..mixins import RandomStateMixin, NumActionsMixin
from ..policy import BasePolicy


__all__ = (
    'VFunction',
    'QFunctionTypeI',
    'QFunctionTypeII',
    'SoftmaxPolicy',
)


class BaseFunctionApproximator(ABC):
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

        if isinstance(self, SoftmaxPolicy):
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

            self._target_model_sync_tau = tau = tf.placeholder(
                tf.float32, shape=())
            self._target_model_sync_op = tf.group(*(
                K.update(wt, wt + tau * (wp - wt))
                for wp, wt in zip(target_weights, primary_weights)))

        K.get_session().run(
            self._target_model_sync_op,
            feed_dict={self._target_model_sync_tau: tau})


class VFunction(BaseFunctionApproximator):
    def __init__(self):
        raise NotImplementedError('VFunction')  # TODO


class BaseQFunction(BaseFunctionApproximator, NumActionsMixin):
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
        assert self.env.observation_space.contains(s)
        assert self.env.action_space.contains(a)
        self._cache.append(s, a, r, done)

        # eager updates
        if self._cache:
            self.batch_update(*self._cache.flush())

    def batch_update(self, S, A, Rn, I_next, S_next, A_next):
        if self.bootstrap_model is not None:
            self.bootstrap_model.train_on_batch(
                [S, Rn, I_next, S_next, A_next], A)
        else:
            G = self.bootstrap_target_np(Rn, I_next, S_next, A_next)
            self.train_model.train_on_batch([S, G], A)

    def bootstrap_target_np(self, Rn, I_next, S_next, A_next=None):
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

        Q_target = Rn + I_next * Q_next
        return Q_target


class QFunctionTypeI(BaseQFunction):

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


class QFunctionTypeII(BaseQFunction):

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


class SoftmaxPolicy(
        BasePolicy, BaseFunctionApproximator, NumActionsMixin,
        RandomStateMixin):

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
        """
        The the next action proposed under the current policy.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a : action, optional

            A single action proposed under the current policy.

        """
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
        assert self.env.observation_space.contains(s)
        assert self.env.action_space.contains(a)
        S = np.expand_dims(s, axis=0)
        A = np.expand_dims(a, axis=0)
        Adv = np.expand_dims(advantage, axis=0)
        self.batch_update(S, A, Adv)

    def batch_eval(self, S, use_target_model=False):
        if use_target_model and self.target_model is not None:
            model = self.target_model
        else:
            model = self.predict_model

        Logits = model.predict_on_batch(S)
        check_numpy_array(Logits, ndim=2, axis_size=self.num_actions, axis=1)
        Pi = softmax(Logits, axis=1)
        return Pi  # shape: [batch_size, num_actions]

    def batch_update(self, S, A, Adv):
        self.train_model.train_on_batch([S, Adv], A)
