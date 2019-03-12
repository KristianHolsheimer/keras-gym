import numpy as np

from ..utils import softmax
from .base import BaseUpdateablePolicy


class GenericSoftmaxPolicy(BaseUpdateablePolicy):
    """
    A generic function approximator for an updateable policy
    :math:`\\hat{\\pi}(a|s)`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    model : keras.Model

        A Keras function approximator. The input shape can be inferred from the
        data, but the output shape must be set to ``[1]``. Check out the
        :mod:`keras_gym.wrappers` module for wrappers that allow you to use
        e.g. scikit-learn function approximators instead.

    random_seed : int, optional

        Set a random state for reproducible randomization.

    """
    MODELTYPE = 2

    def batch_eval(self, X_s):
        dummy_advantages = np.zeros(X_s.shape[0])
        logits = self.model.predict_on_batch([X_s, dummy_advantages])
        assert logits.ndim == 2, "bad shape"  # [batch_size, num_actions]
        proba = softmax(logits, axis=1)
        return proba
