import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ..utils.tensor import check_tensor
from .base import BaseProbaDist


class NormalDist(BaseProbaDist):
    """

    Implementation of a normal distribution.

    Parameters
    ----------
    mu : 1d Tensor, dtype: float, shape: [batch_size, n]

        A batch of vectors of means :math:`\\mu\\in\\mathbb{R}^n`.

    logvar : 1d Tensor, dtype: float, shape: [batch_size, n]

        A batch of vectors of log-variances
        :math:`\\log(\\sigma^2)\\in\\mathbb{R}^n`

    name : str, optional

        Name scope of the distribution.

    random_seed : int, optional

        To get reproducible results.

    """
    PARAM_NAMES = ('mu', 'logvar')

    def __init__(
            self, mu, logvar,
            name='normal_dist',
            random_seed=None):

        check_tensor(mu, ndim=2)
        check_tensor(logvar, same_as=mu)

        self.name = name
        self.mu = mu
        self.logvar = logvar
        self.random_seed = random_seed  # also sets self.random (RandomState)

    def sample(self):
        sigma = K.exp(self.logvar / 2)
        noise = tf.random.normal(
            shape=K.shape(sigma),
            dtype=K.dtype(sigma),
            seed=self.random_seed)

        # reparametrization trick
        x = self.mu + sigma * noise

        return self._rename(x, 'sample')

    def log_proba(self, x):
        check_tensor(x, same_dtype_as=self.mu)
        check_tensor(x, ndim=2)
        check_tensor(x, axis_size=K.int_shape(self.mu)[1], axis=1)

        # abbreviate vars
        m = self.mu
        v = K.exp(self.logvar)
        log_v = self.logvar
        log_2pi = K.constant(np.log(2 * np.pi))

        # main expression, shape: [batch_size, actions_ndim]
        log_p = -0.5 * (log_2pi + log_v + K.square(x - m) / v)

        # aggregate across actions_ndim
        log_p = K.mean(log_p, axis=1)

        return self._rename(log_p, 'log_proba')

    def entropy(self):
        # abbreviate vars
        log_v = self.logvar
        log_2pi = K.constant(np.log(2 * np.pi))

        # main expression
        h = 0.5 * (log_2pi + log_v + 1)

        # aggregate across actions_ndim
        h = K.mean(h, axis=1)

        return self._rename(h, 'entropy')

    def cross_entropy(self, other):
        self._check_other(other)

        # abbreviate vars
        m1 = self.mu
        m2 = other.mu
        v1 = K.exp(self.logvar)
        v2 = K.exp(other.logvar)
        log_v2 = other.logvar
        log_2pi = K.constant(np.log(2 * np.pi))

        # main expression
        ce = 0.5 * (log_2pi + log_v2 + (v1 + K.square(m1 - m2)) / v2)

        # aggregate across actions_ndim
        ce = K.mean(ce, axis=1)

        return self._rename(ce, 'cross_entropy')

    def kl_divergence(self, other):
        self._check_other(other)

        # abbreviate vars
        m1 = self.mu
        m2 = other.mu
        v1 = K.exp(self.logvar)
        v2 = K.exp(other.logvar)
        log_v1 = self.logvar
        log_v2 = other.logvar

        # main expression
        kldiv = 0.5 * (log_v2 - log_v1 + (v1 + K.square(m1 - m2)) / v2 - 1)

        # aggregate across actions_ndim
        kldiv = K.mean(kldiv, axis=1)

        return self._rename(kldiv, 'kl_divergence')
