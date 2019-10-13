from abc import ABC, abstractmethod

from tensorflow.python.keras.losses import Loss


class BaseLoss(ABC, Loss):
    """
    Abstract base class for (stateful) loss functions.

    """
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def from_config(self, *args, **kwargs):
        raise NotImplementedError
    from_config.__doc__ = ""
