import tensorflow as tf
from tensorflow.keras import backend as K

from ..utils import project_onto_actions_tf, check_tensor


__all__ = (
    'SoftmaxPolicyLossWithLogits',
)


class SoftmaxPolicyLossWithLogits:
    """
    Softmax-policy loss (with logits).

    This class provides a stateful implementation of a keras-compatible loss
    function that requires more input than just ``y_true`` and ``y_pred``. The
    required state that this loss function depends on is a batch of so-called
    *advantages* :math:`\\mathcal{A}(s, a)`, which are essentially shifted
    returns, cf. Chapter 13 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. The advantage
    function is often defined as :math:`\\mathcal{A}(s, a) = Q(s, a) - V(s)`.
    The baseline function :math:`V(s)` can be anything you like; a common
    choice is :math:`V(s) = \\sum_a\\pi(a|s)\\,Q(s,a)`, in which case
    :math:`\\mathcal{A}(s, a)` is a proper advantage function with vanishing
    expectation value.

    This loss function is actually a surrogate loss function defined in such a
    way that its gradient is the same as what one would get by taking a true
    policy gradient.

    Parameters
    ----------

    Adv : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one for each time step.

    """
    def __init__(self, Adv):
        check_tensor(Adv, dtype='float')

        if K.ndim(Adv) == 2:
            check_tensor(Adv, axis_size=0, axis=1)
            Adv = K.squeeze(Adv, axis=1)

        check_tensor(Adv, ndim=1)
        self.Adv = K.stop_gradient(Adv)

    @staticmethod
    def logpi_surrogate(logits):
        """
        Construct a surrogate for :math:`\\log\\pi(a|s)` that has the property
        that when we take its gradient it returns the true gradients
        :math:`\\nabla\\log\\pi(a|s)`. In a softmax policy we predict the input
        (or logit) :math:`h(s, a, \\theta)` of the softmax function, such that:

        .. math::

            \\pi_\\theta(a|s)\\ =\\ \\frac
                {\\text{e}^{h_\\theta(s,a)}}
                {\\sum_{a'}\\text{e}^{h_\\theta(s,a')}}

        This means that gradient of the log-policy with respect to the model
        weights :math:`\\theta` is:

        .. math::

            \\nabla\\log\\pi_\\theta(a|s)\\ =\\ \\nabla h_\\theta(s,a)
            - \\sum_{a'}\\pi_\\theta(a'|s)\\,\\nabla h_\\theta(s,a')

        Now this function will actually return the following surrogate for
        :math:`\\log\\pi_\\theta(a|s)`:

        .. math::

            \\texttt{logpi_surrogate}\\ =\\ h_\\theta(s,a) -
            \\sum_{a'}\\texttt{stop_gradient}(\\pi_\\theta(a'|s))\\,
            h_\\theta(s,a')

        This surrogate has the property that its gradient is the same as the
        gradient of :math:`\\log\\pi_\\theta(a|s)`.


        Parameters
        ----------
        logits : 2d Tensor, shape = [batch_size, num_actions]

            The predicted logits of the softmax policy, a.k.a. ``y_pred``.

        Returns
        -------
        logpi_surrogate : Tensor, same shape as input

            The surrogate for :math:`\\log\\pi_\\theta(a|s)`.

        """
        check_tensor(logits, ndim=2)
        pi = K.stop_gradient(K.softmax(logits, axis=1))
        mean_logits = tf.expand_dims(tf.einsum('ij,ij->i', pi, logits), axis=1)
        return logits - mean_logits

    def __call__(self, A, logits):
        """
        Compute the policy-gradient surrogate loss.

        Parameters
        ----------
        A : 2d Tensor, dtype = int, shape = [batch_size, 1]

            This is a batch of actions that were actually taken. This argument
            of the loss function is usually reserved for ``y_true``, i.e. a
            prediction target. In this case, ``A`` doesn't act as a prediction
            target but rather as a mask. We use this mask to project our
            predicted logits down to those for which we actually received a
            feedback signal.

        logits : 2d Tensor, shape = [batch_size, num_actions]

            The predicted logits of the softmax policy, a.k.a. ``y_pred``.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        batch_size = K.int_shape(self.Adv)[0]

        # input shape of A is generally [None, None]
        A.set_shape([None, 1])     # we know that axis=1 must have size 1
        A = tf.squeeze(A, axis=1)  # A.shape = [batch_size]
        A = tf.cast(A, tf.int64)   # must be int (we'll use `A` for slicing)

        # check shapes
        check_tensor(A, ndim=1, axis_size=batch_size, axis=0)
        check_tensor(logits, ndim=2, axis_size=batch_size, axis=0)

        # construct the surrogate for logpi(.|s)
        logpi_all = self.logpi_surrogate(logits)  # [batch_size, num_actions]

        # project onto actions taken: logpi(.|s) --> logpi(a|s)
        logpi = project_onto_actions_tf(logpi_all, A)  # shape: [batch_size]

        # construct the final surrogate loss
        surrogate_loss = -K.mean(self.Adv * logpi)

        return surrogate_loss
