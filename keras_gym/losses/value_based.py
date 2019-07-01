import tensorflow as tf
from tensorflow.keras import backend as K

from ..utils import project_onto_actions_tf, check_tensor
from .base import BaseLoss

__all__ = (
    'Huber',
    'ProjectedSemiGradientLoss',
    'RootMeanSquaredError',
    'LoglossSign',
)


class Huber(BaseLoss):
    """
    Huber loss that is compatible with both tensorflow 1.x and 2.x.

    Parameters
    ----------
    delta : float

        The point where the Huber loss function changes from a quadratic to
        linear.

    name : str, optional

        Optional name for the op.

    """
    def __init__(self, delta=1.0, name='huber_loss'):
        self.delta = delta
        self._name = name
        if tf.__version__ >= '2.0':
            self._func = tf.keras.losses.Huber(delta=delta, name=name)
        else:
            self._func = None

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Compute the Huber loss.

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
        if self._func is not None:
            return self._func(y_true, y_pred, sample_weight=sample_weight)
        else:
            return tf.losses.huber_loss(
                y_true, y_pred, delta=self.delta, scope=self._name)


class RootMeanSquaredError(BaseLoss):
    """
    Root-mean-squared error (RMSE) loss.

    Parameters
    ----------
    name : str, optional

        Optional name for the op.

    """
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
    def __init__(self, G, base_loss=Huber()):
        check_tensor(G)

        if K.ndim(G) == 2:
            check_tensor(G, axis_size=1, axis=1)
            G = K.squeeze(G, axis=1)

        check_tensor(G, ndim=1)
        self.G = K.stop_gradient(G)
        self.base_loss = base_loss

    def __call__(self, P, Q_pred, sample_weight=None):
        """
        Compute the projected MSE.

        Parameters
        ----------
        P : 2d Tensor, dtype: int, shape: [batch_size, num_actions]

            A batch of action propensities, a.k.a. ``y_true``. In a typical
            application, :term:`P` is just an indicator for which action was
            chosen by the behavior policy. In this sense, :term:`P` acts as a
            projector more than a prediction target. That is, :term:`P` is used
            to project our predicted values down to those for which we actually
            received the feedback signal: :term:`G`.

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
        # check shapes
        batch_size = K.int_shape(self.G)[0]
        check_tensor(P, ndim=2, axis_size=batch_size, axis=0)
        check_tensor(Q_pred, ndim=2, axis_size=batch_size, axis=0)

        # project onto actions taken: q(s,.) --> sum_a pi(a|s) q(s,a)
        Q_pred_projected = tf.einsum('ij,ij->i', Q_pred, P)

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
