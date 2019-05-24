import tensorflow as tf
from tensorflow.keras import backend as K

from ..utils import project_onto_actions_tf, check_tensor


__all__ = (
    'ProjectedSemiGradientLoss',
)


class ProjectedSemiGradientLoss:
    """
    Loss function for type-II Q-function.

    This loss function projects the predictions :math:`Q(s, .)` onto the
    actions for which we actually received a feedback signal.

    Parameters
    ----------
    G : 1d Tensor, dtype: float, shape: [batch_size]

        The returns that we wish to fit our value function on.

    base_loss : keras loss function, optional

        Keras loss function. Default: :func:`huber_loss
        <tensorflow.losses.huber_loss>`.

    """
    def __init__(self, G, base_loss=tf.losses.huber_loss):
        check_tensor(G)

        if K.ndim(G) == 2:
            check_tensor(G, axis_size=1, axis=1)
            G = K.squeeze(G, axis=1)

        check_tensor(G, ndim=1)
        self.G = K.stop_gradient(G)
        self.base_loss = base_loss

    def __call__(self, A, Q_pred):
        """
        Compute the projected MSE.

        Parameters
        ----------
        A : 2d Tensor, dtype = int, shape = [batch_size, 1]

            This is a batch of actions that were actually taken. This argument
            of the loss function is usually reserved for ``y_true``, i.e. a
            prediction target. In this case, ``A`` doesn't act as a prediction
            target but rather as a mask. We use this mask to project our
            predicted logits down to those for which we actually received a
            feedback signal.

        Q_pred : 2d Tensor, shape = [batch_size, num_actions]

            The predicted values :math:`Q(s,.)`, a.k.a. ``y_pred``.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        # input shape of A is generally [None, None]
        check_tensor(A, ndim=2)
        A.set_shape([None, 1])     # we know that axis=1 must have size 1
        A = tf.squeeze(A, axis=1)  # A.shape = [batch_size]
        A = tf.cast(A, tf.int64)   # must be int (we'll use `A` for slicing)

        # check shapes
        batch_size = K.int_shape(self.G)[0]
        check_tensor(A, ndim=1, axis_size=batch_size, axis=0)
        check_tensor(Q_pred, ndim=2, axis_size=batch_size, axis=0)

        # project onto actions taken: Q(s,.) --> Q(s,a)
        Q_pred_projected = project_onto_actions_tf(Q_pred, A)

        # the actual loss
        return self.base_loss(self.G, Q_pred_projected)
