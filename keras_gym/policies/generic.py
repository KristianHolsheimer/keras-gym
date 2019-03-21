import numpy as np

from ..utils import softmax
from .base import BaseUpdateablePolicy, BaseActorCritic


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
    @property
    def output_dim(self):
        return self.num_actions  # error if action space is not discrete

    def batch_eval(self, X_s):
        dummy_advantages = np.zeros(X_s.shape[0])
        logits = self.model.predict_on_batch([X_s, dummy_advantages])
        assert logits.ndim == 2, "bad shape"  # [batch_size, num_actions]
        proba = softmax(logits, axis=1)
        return proba


class GenericActorCritic(BaseActorCritic):
    """
    This is a simple wrapper class that combines a policy (actor) with a
    value function (critic) into a sigle object.

    We don't strictly need this, as our actor-critic type algorithms can take
    the policy and value function as separate arguments without an issue. There
    are situations, however, in which it is very useful to to have the policy
    and value function packaged together.

    TODO: class signature

    """
    pass
