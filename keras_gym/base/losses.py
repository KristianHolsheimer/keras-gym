from abc import ABC, abstractmethod

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import Loss
from ..utils import check_tensor


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


class BasePolicyLoss(BaseLoss):
    """
    Abstract base class for policy gradient loss functions.

    This base class provides the ``set_advantage(Adv)`` method.

    Parameters
    ----------

    Adv : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one for each time step.

    """
    def __init__(self, Adv):
        self.set_advantage(Adv)

    def set_advantage(self, Adv):
        """
        Set the :term:`Adv` tensor of the stateful policy-gradient loss
        function.

        Parameters
        ----------

        Adv : 1d Tensor, dtype: float, shape: [batch_size]

            The advantages, one for each time step.

        Returns
        -------
        self

            The updated instance.

        """
        check_tensor(Adv, dtype='float')

        if K.ndim(Adv) == 2:
            check_tensor(Adv, axis_size=1, axis=1)
            Adv = K.squeeze(Adv, axis=1)

        check_tensor(Adv, ndim=1)
        self.Adv = K.stop_gradient(Adv)
        return self
