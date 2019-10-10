import numpy as np
import tensorflow as tf
import gym
from scipy import stats
from scipy.special import expit as sigmoid

from ..utils.helpers import check_tensor
from ..base.errors import ActionSpaceError
from .base import BaseProbaDist


class NormalDist(BaseProbaDist):
    """

    Implementation of a (squashed) normal distribution.

    Although the sampled values live on a bounded :class:`Box <gym.spaces.Box>`
    space, the underlying distribution does describe a bona fide normal
    distribution. It's only after sampling from the normal distribution that
    the resulting value is squashed down to the bounded :class:`Box
    <gym.spaces.Box>` space.

    Parameters
    ----------
    space : gym.spaces.Box

        A :class:`Box <gym.spaces.Box>` space.

    mu : 1d Tensor, dtype: float, shape: [batch_size, actions_ndim]

        A batch of vectors of means :math:`\\mu\\in\\mathbb{R}^n`, where
        :math:`n` is the dimensionlity of the :class:`Box <gym.spaces.Box>`
        action space.

    logvar : 1d Tensor, dtype: float, shape: [batch_size, actions_ndim]

        A batch of vectors of log-variances
        :math:`\\log(\\sigma^2)\\in\\mathbb{R}^n`, where :math:`n` is the
        dimensionlity of the :class:`Box <gym.spaces.Box>` action space.

    TODO: fix docstring

    """
    def __init__(self, space, model, name='', random_seed=None):
        if not isinstance(space, gym.spaces.Box):
            raise ActionSpaceError("space must be a Box")

        self.name = name
        self.space = space
        self.ndim = len(space.shape)
        self.random_seed = random_seed  # also sets self.random (RandomState)

        # this will set attrs: model, mu, logvar
        self._check_model_and_set_params(model)

    def sample(self, inputs_np):
        mu_np, logvar_np = self.model.predict_on_batch(inputs_np)
        sigma_np = np.exp(0.5 * logvar_np)
        z = stats.norm(loc=mu_np, scale=sigma_np).rvs(random_state=self.random)
        x = self.space.low + (self.space.high - self.space.low) * sigmoid(z)
        return x

    def log_proba(self, x):
        check_tensor(x, same_as=self.mu)
        z = (x - self.space.low) / (self.space.high - self.space.low)

        # abbreviate vars
        m = self.mu
        v = tf.exp(self.logvar)
        log_v = self.logvar
        log_2pi = tf.constant(np.log(2 * np.pi))

        # main expression
        log_p = -0.5 * (log_2pi + log_v + tf.square(z - m) / v)

        return tf.no_op(log_p, name=f'{self.name}/normal_dist/log_proba')

    def entropy(self):
        # abbreviate vars
        log_v = self.logvar
        log_2pi = tf.constant(np.log(2 * np.pi))

        # main expression
        h = 0.5 * (log_2pi + log_v + 1)

        return tf.no_op(h, name=f'{self.name}/normal_dist/entropy')

    def cross_entropy(self, other):
        self._check_other(other)

        # abbreviate vars
        m1 = self.mu
        m2 = other.mu
        v1 = tf.exp(self.logvar)
        v2 = tf.exp(other.logvar)
        log_v2 = other.logvar
        log_2pi = tf.constant(np.log(2 * np.pi))

        # main expression
        ce = 0.5 * (log_2pi + log_v2 + (v1 + tf.square(m1 - m2)) / v2)

        return tf.no_op(ce, name=f'{self.name}/normal_dist/cross_entropy')

    def kl_divergence(self, other):
        self._check_other(other)

        # abbreviate vars
        m1 = self.mu
        m2 = other.mu
        v1 = tf.exp(self.logvar)
        v2 = tf.exp(other.logvar)
        log_v1 = self.logvar
        log_v2 = other.logvar

        # main expression
        kldiv = 0.5 * (log_v2 - log_v1 + (v1 + tf.square(m1 - m2)) / v2 - 1)

        return tf.no_op(kldiv, name=f'{self.name}/normal_dist/kl_divergence')

    def proba_ratio(self, other, x):
        rho = tf.exp(self.log_proba(x) - other.log_proba(x))
        return tf.no_op(rho, name=f'{self.name}/normal_dist/proba_ratio')

    def _check_model(self, model):
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"expected a keras.Model, got: {model.__class__.__name__}")
        if len(model.outputs) != 2:
            raise ValueError(
                "expected a model with two outputs (mu, logvar), the "
                f"provided model has {len(model.outputs)} outputs instead")
        if not model.output_names[0].endswith('/mu'):
            raise ValueError(r"the first output must have name {scope}/mu")
        if not model.output_names[1].endswith('/logvar'):
            raise ValueError(r"the first output must have name {scope}/logvar")
        mu, logvar = model.outputs
        check_tensor(mu, ndim=2, axis_size=self.ndim, axis=1)
        check_tensor(logvar, same_as=mu)
        self.mu = mu
        self.logvar = logvar
        self.model = model
