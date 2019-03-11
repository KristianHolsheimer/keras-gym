import numpy as np

from ..utils import softmax
from .base import BaseUpdateablePolicy


class SoftmaxPolicy(BaseUpdateablePolicy):
    MODELTYPE = 2

    def batch_eval(self, X_s):
        dummy_advantages = np.zeros(X_s.shape[0])
        logits = self.model.predict_on_batch([X_s, dummy_advantages])
        assert logits.ndim == 2, "bad shape"  # [batch_size, num_actions]
        proba = softmax(logits, axis=1)
        return proba
