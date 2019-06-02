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

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    """
    def __init__(self, Adv, entropy_bonus=0.01):
        self.entropy_bonus = float(entropy_bonus)
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

    @abstractmethod
    def __call__(self, A, Z, sample_weight):
        """
        Compute the policy-gradient surrogate loss.

        Parameters
        ----------
        A : 2d Tensor, dtype: int, shape: [batch_size, 1]

            This is a batch of actions that were actually taken. This argument
            of the loss function is usually reserved for ``y_true``, i.e. a
            prediction target. In this case, ``A`` doesn't act as a prediction
            target but rather as a mask. We use this mask to project our
            predicted values down to those for which we actually received a
            feedback signal.

        Z : 2d Tensor, shape: [batch_size, num_actions]

            The predicted logits of the softmax policy, a.k.a. ``y_pred``.

        sample_weight : 1d Tensor, dtype: float, shape: [batch_size], optional

            Not yet implemented; will be ignored.

            #TODO: implement this -Kris

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        pass
