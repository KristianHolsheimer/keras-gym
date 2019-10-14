from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ..base.mixins import RandomStateMixin, ActionSpaceMixin, LoggerMixin
from ..utils import check_tensor
from ..policies.base import BasePolicy


__all__ = (
    'BaseFunctionApproximator',
    'BaseUpdateablePolicy',
)


class BaseFunctionApproximator(ABC, LoggerMixin, ActionSpaceMixin, RandomStateMixin):  # noqa: E501
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

    def _check_attrs(self, skip=None):
        required_attrs = [
            'env', 'gamma', 'bootstrap_n', 'train_model', 'predict_model',
            'target_model', '_cache']

        if skip is None:
            skip = []

        missing_attrs = ", ".join(
            attr for attr in required_attrs
            if attr not in skip and not hasattr(self, attr))

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
        if tau > 1 or tau < 0:
            ValueError("tau must lie on the unit interval [0,1]")

        for m in ('model', 'param_model', 'greedy_model'):
            if hasattr(self, 'target_' + m):
                p = getattr(self, 'predict_' + m)
                t = getattr(self, 'target_' + m)
                Wp = p.get_weights()
                Wt = t.get_weights()
                Wt = [wt + tau * (wp - wt) for wt, wp in zip(Wt, Wp)]
                t.set_weights(Wt)


class BaseUpdateablePolicy(BasePolicy, BaseFunctionApproximator):
    """
    Base class for modeling :term:`updateable policies <updateable policy>`.

    Parameters
    ----------
    function_approximator : FunctionApproximator

        The main :class:`FunctionApproximator <keras_gym.FunctionApproximator>`
        object.

    update_strategy : str, callable, optional

        The strategy for updating our policy. This determines the loss function
        that we use for our policy function approximator. If you wish to use a
        custom policy loss, you can override the
        :func:`policy_loss_with_metrics` method.

        Provided options are:

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

    ppo_clip_eps : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_beta : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    UPDATE_STRATEGIES = ('vanilla', 'ppo', 'cross_entropy')

    def __init__(
            self, function_approximator,
            update_strategy='vanilla',
            ppo_clip_eps=0.2,
            entropy_beta=0.01,
            random_seed=None):

        self.function_approximator = function_approximator
        self.env = self.function_approximator.env
        self.update_strategy = update_strategy
        self.ppo_clip_eps = float(ppo_clip_eps)
        self.entropy_beta = float(entropy_beta)
        self.random_seed = random_seed  # sets self.random via RandomStateMixin

        self._init_models()
        self._check_attrs()

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
        S = np.expand_dims(s, axis=0)
        A = self.batch_eval(S, use_target_model)
        return A[0]

    def dist_params(self, s, use_target_model=False):
        """

        Get the parameters of the (conditional) probability distribution
        :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        use_target_model : bool, optional

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        \\*params : tuple of arrays

            The raw distribution parameters.

        """
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        if use_target_model:
            params = self.target_param_model.predict(S)
        else:
            params = self.predict_param_model.predict(S)

        # extract single instance
        if isinstance(params, list):
            params = [arr[0] for arr in params]
        elif isinstance(params, np.ndarray):
            params = params[0]
        else:
            TypeError(f"params have unexpected type: {type(params)}")

        return params

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
        assert self.env.observation_space.contains(s)
        S = np.expand_dims(s, axis=0)
        if use_target_model:
            A = self.target_greedy_model.predict(S)
        else:
            A = self.predict_greedy_model.predict(S)
        return A[0]

    def update(self, s, a, advantage):
        """
        Update the policy.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action.

        advantage : float

            A value for the advantage :math:`\\mathcal{A}(s,a) = q(s,a) -
            v(s)`. This might be sampled and/or estimated version of the true
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

            Whether to use the :term:`target_model` internally. If False
            (default), the :term:`predict_model` is used.

        Returns
        -------
        A : 2d array, shape: [batch_size]

            A batch of sampled actions.

        """
        if use_target_model:
            A = self.target_model.predict(S)
        else:
            A = self.predict_model.predict(S)
        return A

    def batch_update(self, S, A, Adv):
        """
        Update the policy on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : nd array, shape: [batch_size, ...]

            A batch of actions taken by the behavior policy.

        Adv : 1d array, dtype: float, shape: [batch_size]

            A value for the :term:`advantage <Adv>` :math:`\\mathcal{A}(s,a) =
            q(s,a) - v(s)`. This might be sampled and/or estimated version of
            the true advantage.

        Returns
        -------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        losses = self._train_on_batch([S, A, Adv], None)
        return losses

    def policy_loss_with_metrics(self, Adv, A):
        """

        This method constructs the policy loss as a scalar-valued Tensor,
        together with a dictionary of metrics (also scalars).

        This method may be overridden to construct a custom policy loss and/or
        to change the accompanying metrics.

        Parameters
        ----------
        Adv : 1d Tensor, shape: [batch_size]

            A batch of advantages.

        A : nd Tensor, shape: [batch_size, ...]

            A batch of actions taken under the behavior policy.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The policy loss. This can be fed to a keras Model using
            ``model.add_loss(loss)``.


        """
        Adv = K.stop_gradient(Adv)
        if K.ndim(Adv) == 2:
            Adv = K.squeeze(Adv, axis=1)
        check_tensor(Adv, ndim=1)

        if self.update_strategy == 'vanilla':

            log_pi = self.dist.log_proba(A)
            check_tensor(log_pi, same_as=Adv)

            entropy = K.mean(self.dist.entropy())

            # flip sign to get loss from objective
            loss = -K.mean(Adv * log_pi) + self.entropy_beta * entropy

            # no metrics related to behavior_dist since its not used in loss
            metrics = {'policy/entropy': entropy}

        elif self.update_strategy == 'ppo':

            log_pi = self.dist.log_proba(A)
            log_pi_old = K.stop_gradient(self.target_dist.log_proba(A))
            check_tensor(log_pi, same_as=Adv)
            check_tensor(log_pi_old, same_as=Adv)

            eps = self.ppo_clip_eps
            ratio = K.exp(log_pi - log_pi_old)
            ratio_clip = K.clip(ratio, 1 - eps, 1 + eps)
            check_tensor(log_pi, same_as=Adv)
            check_tensor(log_pi_old, same_as=Adv)

            clip_objective = K.mean(K.minimum(Adv * ratio, Adv * ratio_clip))
            entropy = K.mean(self.dist.entropy())
            kl_div = K.mean(self.target_dist.kl_divergence(self.dist))

            # flip sign to get loss from objective
            loss = -(clip_objective + self.entropy_beta * entropy)
            metrics = {'policy/entropy': entropy, 'policy/kl_div': kl_div}

        elif self.update_strategy == 'cross_entropy':
            raise NotImplementedError('cross_entropy')

        else:
            raise ValueError(
                "unknown update_strategy '{}'".format(self.update_strategy))

        # rename
        loss = tf.identity(loss, name='policy_loss')

        return loss, metrics

    def _check_attrs(self):
        model = [
            'predict_model',
            'target_model',
            'predict_greedy_model',
            'target_greedy_model',
            'predict_param_model',
            'target_param_model',
            'train_model',
        ]
        misc = [
            'function_approximator',
            'env',
            'update_strategy',
            'ppo_clip_eps',
            'entropy_beta',
            'random_seed',
        ]
        missing_attrs = [a for a in model + misc if not hasattr(self, a)]
        if missing_attrs:
            raise AttributeError(
                "missing attributes: {}".format(", ".join(missing_attrs)))
