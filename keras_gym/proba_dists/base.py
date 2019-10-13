from abc import ABC, abstractmethod

import tensorflow as tf

from ..base.mixins import RandomStateMixin


__all__ = (
    'BaseProbaDist',
)


class BaseProbaDist(ABC, RandomStateMixin):
    @abstractmethod
    def __init__(self, env, *params, **settings):
        pass

    @abstractmethod
    def sample(self):
        """
        Sample from the probability distribution.

        Returns
        -------
        sample : 1d Tensor, shape: [batch_size, \\*variate_shape]

            The sampled variates.

        """
        pass

    @abstractmethod
    def log_proba(self, x):
        """
        Compute the log-probability associated with specific variates.

        Parameters
        ----------
        x : nd Tensor, shape: [batch_size, ...]

            A batch of specific variates.

        Returns
        -------
        log_proba : 1d Tensor, shape: [batch_size]

            The log-probabilities.

        """
        pass

    @abstractmethod
    def entropy(self):
        """
        Compute the entropy of the probability distribution.

        Parameters
        ----------
        x : nd Tensor, shape: [batch_size, ...]

            A batch of specific variates.

        Returns
        -------
        entropy : 1d Tensor, shape: [batch_size]

            The entropy of the probability distribution.

        """
        pass

    @abstractmethod
    def cross_entropy(self, other):
        """

        Compute the cross-entropy of a probability distribution
        :math:`p_\\text{other}` relative to the current probablity
        distribution :math:`p_\\text{self}`, symbolically:

        .. math::

            \\text{CE}[p_\\text{self}, p_\\text{other}]\\ =\\
                -\\sum p_\\text{self}\\,\\log p_\\text{other}

        Parameters
        ----------
        other : probability dist

            The ``other`` probability dist must be of the same type as
            ``self``.

        Returns
        -------
        cross_entropy : 1d Tensor, shape: [batch_size]

            The cross-entropy.

        """
        pass

    def kl_divergence(self, other):
        """

        Compute the Kullback-Leibler divergence of a probability distribution
        :math:`p_\\text{other}` relative to the current probablity
        distribution :math:`p_\\text{self}`, symbolically:

        .. math::

            \\text{KL}[p_\\text{self}, p_\\text{other}]\\ =\\
                -\\sum p_\\text{self}\\,
                    \\log\\frac{p_\\text{other}}{p_\\text{self}}

        Parameters
        ----------
        other : probability dist

            The ``other`` probability dist must be of the same type as
            ``self``.

        Returns
        -------
        kl_divergence : 1d Tensor, shape: [batch_size]

            The KL-divergence.

        """
        return self.cross_entropy(other) - self.entropy()

    def _rename(self, tensor, name):
        return tf.identity(tensor, f'{self.name}/{name}')

    def _check_other(self, other):
        if type(other) is not type(self):
            raise TypeError("'other' must be of the same type as 'self'")
