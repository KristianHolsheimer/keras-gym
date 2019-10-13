import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import check_tensor
from .base import BaseLoss

__all__ = (
    'ProjectedSemiGradientLoss',
    'RootMeanSquaredError',
    'LoglossSign',
)


class RootMeanSquaredError(BaseLoss):
    """
    Root-mean-squared error (RMSE) loss.

    Parameters
    ----------
    name : str, optional

        Optional name for the op.

    """
    name = 'rmse'

    def __init__(self, delta=1.0, name='root_mean_squared_error'):
        self._func = tf.keras.losses.MeanSquaredError(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Compute the RMSE loss.

        Parameters
        ----------
        y_true : Tensor, shape: [batch_size, ...]

            Ground truth values.

        y_pred : Tensor, shape: [batch_size, ...]

            The predicted values.

        sample_weight : Tensor, dtype: float, optional

            Tensor whose rank is either 0, or the same rank as ``y_true``, or
            is broadcastable to ``y_true``. ``sample_weight`` acts as a
            coefficient for the loss. If a scalar is provided, then the loss is
            simply scaled by the given value. If ``sample_weight`` is a tensor
            of size ``[batch_size]``, then the total loss for each sample of
            the batch is rescaled by the corresponding element in the
            ``sample_weight`` vector. If the shape of sample_weight matches the
            shape of ``y_pred``, then the loss of each measurable element of
            ``y_pred`` is scaled by the corresponding value of
            ``sample_weight``.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        return K.sqrt(self._func(y_true, y_pred, sample_weight=sample_weight))


class ProjectedSemiGradientLoss(BaseLoss):
    """
    Loss function for type-II Q-function.

    This loss function projects the predictions :math:`q(s, .)` onto the
    actions for which we actually received a feedback signal.

    Parameters
    ----------
    G : 1d Tensor, dtype: float, shape: [batch_size]

        The returns that we wish to fit our value function on.

    base_loss : keras loss function, optional

        Keras loss function. Default: :func:`huber_loss
        <tensorflow.losses.huber_loss>`.

    """
    def __init__(self, G, base_loss=keras.losses.Huber()):
        check_tensor(G)

        if K.ndim(G) == 2:
            check_tensor(G, axis_size=1, axis=1)
            G = K.squeeze(G, axis=1)

        check_tensor(G, ndim=1)
        self.G = K.stop_gradient(G)
        self.base_loss = base_loss

    def __call__(self, A, Q_pred, sample_weight=None):
        """
        Compute the projected MSE.

        Parameters
        ----------
        A : 2d Tensor, dtype: int, shape: [batch_size, num_actions]

            A batch of (one-hot encoded) discrete actions :term:`A`.

        Q_pred : 2d Tensor, shape: [batch_size, num_actions]

            The predicted values :math:`q(s,.)`, a.k.a. ``y_pred``.

        sample_weight : Tensor, dtype: float, optional

            Tensor whose rank is either 0 or is broadcastable to ``y_true``.
            ``sample_weight`` acts as a coefficient for the loss. If a scalar
            is provided, then the loss is simply scaled by the given value. If
            ``sample_weight`` is a tensor of size ``[batch_size]``, then the
            total loss for each sample of the batch is rescaled by the
            corresponding element in the ``sample_weight`` vector.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        # check/fix shapes and dtypes
        batch_size = K.int_shape(self.G)[0]
        check_tensor(Q_pred, ndim=2, axis_size=batch_size, axis=0)
        check_tensor(A, ndim=2, axis_size=batch_size, axis=0)
        A.set_shape(K.int_shape(Q_pred))

        # project onto actions taken: q(s,.) --> q(s,a)
        Q_pred_projected = tf.einsum('ij,ij->i', Q_pred, A)

        # the actual loss
        return self.base_loss(
            self.G, Q_pred_projected, sample_weight=sample_weight)


class LoglossSign(BaseLoss):
    """
    Logloss implemented for predicted logits :math:`z\\in\\mathbb{R}` and
    ground truth :math:`y\\pm1`.

    .. math::

        L\\ =\\ \\log\\left( 1 + \\exp(-y\\,z) \\right)

    """
    def __init__(self):
        pass

    def __call__(self, y_true, z_pred, sample_weight=None):
        """
        Parameters
        ----------
        y_true : Tensor, shape: [batch_size, ...]

            Ground truth values :math:`y\\pm1`.

        z_pred : Tensor, shape: [batch_size, ...]

            The predicted logits :math:`z\\in\\mathbb{R}`.

        sample_weight : Tensor, dtype: float, optional

            Not yet implemented.

            #TODO: implement this

        """
        if K.dtype(z_pred) == 'float32':
            z_pred = K.clip(z_pred, -15, 15)
        elif K.dtype(z_pred) == 'float64':
            z_pred = K.clip(z_pred, -30, 30)
        else:
            raise TypeError('Expected dtype for z_pred: float32 or float64')

        return K.log(1 + K.exp(-y_true * z_pred))
