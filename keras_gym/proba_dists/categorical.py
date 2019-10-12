import tensorflow as tf
from tensorflow.keras import backend as K

from ..utils.tensor import log_softmax_tf, check_tensor

from .base import BaseProbaDist


class CategoricalDist(BaseProbaDist):
    """
    Implementation of a categorical distribution.

    Parameters
    ----------
    logits : 2d Tensor, dtype: float, shape: [batch_size, num_categories]

        A batch of logits :math:`z\\in \\mathbb{R}^n` with :math:`n=`
        ``num_categories``.

    boltzmann_tau : float, optional

        The Boltzmann temperature that is used in generating near one-hot
        propensities in :func:`sample`. A smaller number means closer to
        deterministic, one-hot encoded samples. A larger number means better
        numerical stability. A good value for :math:`\\tau` is one that offers
        a good trade-off between these two desired properties.

    name : str, optional

        Name scope of the distribution.

    random_seed : int, optional

        To get reproducible results.

    """
    def __init__(
            self, logits,
            boltzmann_tau=0.2,
            name='categorical_dist',
            random_seed=None):

        check_tensor(logits, ndim=2)
        self.num_categories = K.int_shape(logits)[1]

        self.name = name
        self.logits = logits
        self.boltzmann_tau = boltzmann_tau
        self.random_seed = random_seed  # also sets self.random (RandomState)

    def sample(self, n=1):
        """

        Sample from the probability distribution.

        .. note::

            The :func:`CategoricalDist.sample
            <keras_gym.proba_dists.CategoricalDist.sample>` has not yet been
            implemented in a differentiable way, but it will use the approach
            outlined in `[ArXiv:1611.01144]
            <https://arxiv.org/abs/1611.01144>`_

        Returns
        -------
        sample : 2d array, shape: [batch_size, num_categories]

            The sampled variates. The returned arrays are near one-hot encoded
            versions of deterministic variates.

        """
        # FIXME: write differentiable implementation
        return K.one_hot(K.argmax(self.logits), self.num_categories)

    def log_proba(self, x):
        if K.ndim(x) == 2 and K.int_shape(x)[1] == 1:
            x = K.squeeze(x, axis=1)
        if K.ndim(x) == 1:
            x = K.one_hot(x, self.num_categories)
        check_tensor(x, same_as=self.logits)

        logp = tf.einsum('ij,ij->i', x, log_softmax_tf(self.logits))

        return self._rename(logp, 'log_proba')

    def entropy(self):
        p = K.softmax(self.logits)
        logp = log_softmax_tf(self.logits)
        h = tf.einsum('ij,ij->i', p, -logp)
        return self._rename(h, 'entropy')

    def cross_entropy(self, other):
        self._check_other(other)
        p_self = K.softmax(self.logits)
        logp_other = log_softmax_tf(other.logits)
        ce = tf.einsum('ij,ij->i', p_self, -logp_other)
        return self._rename(ce, 'cross_entropy')

    def kl_divergence(self, other):
        self._check_other(other)
        p_self = K.softmax(self.logits)
        logp_self = log_softmax_tf(self.logits)
        logp_other = log_softmax_tf(other.logits)
        kl_div = tf.einsum('ij,ij->i', p_self, logp_self - logp_other)
        return self._rename(kl_div, 'kl_divergence')
