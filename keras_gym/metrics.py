import tensorflow as tf
from tensorflow.keras import backend as K


def masked_mse_loss(y_true, y_pred):
    """
    Custom MSE loss that excludes zeros from the mean. This is particularly
    useful when training a type-II model, for which `y_true` is the same as
    `y_pred` but for one entry per sample. For instance, let's say we have
    an environment with `num_actions=4` and suppose we have a data point
    that was generated by sampling a specific action `a=2`. In that case,
    `y_true` and `y_pred` may look like:

        y_pred = [3.1, -1.2, 0.1, 7.4]
        y_true = [3.1, -1.2, 9.8, 7.4]

    Since only the `a=2` entry received an update, the only new information
    is in the residual `y_pred[2] - y_true[2]`. All of the other residuals
    are trivially zero.

    This loss function ignores the residuals that are trivially zero by
    applying an automated mask.

    .. note:: This function requires the Tensorflow backend.

    """

    # create mask
    eps = 1e-10
    err = K.flatten(y_pred - y_true)
    mask = K.greater(K.abs(err), eps)

    # compute masked MSE
    masked_err = tf.boolean_mask(err, mask)
    mse = K.mean(K.square(masked_err))

    return mse


class SoftmaxPolicyLossWithLogits:
    """
    This class provides a stateful implementation of a keras-compatible loss
    function that requires more input than just `y_true` and `y_pred`. The
    required state that this loss function depends on is a batch of so-called
    `advantages`, which are essentially returns that are centered around 0, cf.
    Chapter 13 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    This loss function is actually a surrogate loss function defined in such a
    way that its gradient is the same as what one would get by taking a true
    policy gradient.

    Parameters
    ----------
    advantages : 2d-Tensor, shape: [batch_size]
        A tensor that contains the advantages, i.e.
        :math:`\\mathcal{A}(s, a) = Q(s, a) - V(s)`. The baseline function
        :math:`V(s)` can be anything you like; a common choice is
        :math:`V(s) = \\sum_a\\pi(a|s)\\,Q(s,a)`.


    .. note:: This loss function requires tensorflow backend.

    """
    def __init__(self, advantages):
        if K.ndim(advantages) == 2:
            assert K.int_shape(advantages)[1] == 1, "bad shape"
            advantages = K.squeeze(advantages, axis=1)
        assert K.ndim(advantages) == 1, "bad shape"
        self.advantages = K.stop_gradient(advantages)

    def __call__(self, A, logits):
        """
        Compute the policy-gradient surrogate loss.

        Parameters
        ----------
        A : 2d-Tensor, dtype = int, shape = [batch_size, 1]
            This is a batch of actions that were actually taken. This argument
            of the loss function is usually reserved for `y_true`, i.e. a
            prediction target. In this case, `A` doesn't act as a prediction
            target but rather as a mask. We use this mask to project our
            predicted logits down to those for which we actually received a
            feedback signal.

        logits : 2d-Tensor, shape = [batch_size, num_actions]
            The predicted logits of the softmax policy, a.k.a. `y_pred`.

        """
        adv = self.advantages  # this is why loss function is stateful

        # input shape of A is generally [None, None]
        A.set_shape([None, 1])     # we know that axis=1 must have size 1
        A = tf.squeeze(A, axis=1)  # A.shape = [batch_size]
        A = tf.cast(A, tf.int64)   # must be int (we'll use `A` for slicing)

        # check shapes
        assert K.ndim(A) == 1, "bad shape"
        assert K.ndim(logits) == 2, "bad shape"
        assert K.ndim(adv) == 1, "bad shape"
        assert K.int_shape(adv) == K.int_shape(A), "bad shape"
        assert K.int_shape(adv)[0] == K.int_shape(logits)[0], "bad shape"

        # construct the surrogate for logpi(.|s)
        # *note* This surrogate is constructed in such a way that
        #        its gradient is the same as that of logpi(.|s),
        #        hence the stop_gradient in 'pi'.
        pi = K.stop_gradient(K.softmax(logits, axis=1))
        mean_logits = tf.expand_dims(
            tf.einsum('ij,ij->i', logits, pi), axis=1)
        logpi_all = logits - mean_logits  # shape: [batch_size, n_actions]

        # project onto actions taken: logpi(.|s) --> logpi(a|s)
        # *note* Please let me know if there's a better way to do this.
        batch_size = tf.cast(K.shape(A)[0], tf.int64)
        idx = tf.range(batch_size, dtype=A.dtype)
        indices = tf.stack([idx, A], axis=1)
        logpi = tf.gather_nd(logpi_all, indices)  # shape: [batch_size]

        # construct the final surrogate loss
        surrogate_loss = -K.mean(adv * logpi)

        return surrogate_loss