from abc import ABC, abstractmethod


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .utils import project_onto_actions


class BasePolicyLoss(ABC):
    """
    Abstract base class for policy loss functions.

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

    Parameters
    ----------

    advantages : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one per time step.

    """
    def __init__(self, advantages):
        if K.ndim(advantages) == 2:
            assert K.int_shape(advantages)[1] == 1, "bad shape"
            advantages = K.squeeze(advantages, axis=1)
        assert K.ndim(advantages) == 1, "bad shape"
        self.advantages = K.stop_gradient(advantages)

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class SoftmaxPolicyLossWithLogits(BasePolicyLoss):
    """
    Softmax-policy loss (with logits).

    This loss function is actually a surrogate loss function defined in such a
    way that its gradient is the same as what one would get by taking a true
    policy gradient.

    Parameters
    ----------
    advantages : 1d Tensor, dtype: float, shape: [batch_size]
        The advantages, one per time step.

    """
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
        assert K.ndim(logits) == 2, "bad shape"
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
        logpi_all = self.logpi_surrogate(logits)  # [batch_size, num_actions]

        # project onto actions taken: logpi(.|s) --> logpi(a|s)
        logpi = project_onto_actions(logpi_all, A)  # shape: [batch_size]

        # construct the final surrogate loss
        surrogate_loss = -K.mean(adv * logpi)

        return surrogate_loss


class QTypeIIMeanSquaredErrorLoss:
    """
    Loss function for type-II Q-function.

    This loss function projects teh predictions :math:`Q(s, .)` onto the
    actions for which we actually received a feedback signal.

    Parameters
    ----------
    G : 1d Tensor, dtype: float, shape: [batch_size]

        The returns that we wish to fit our value function on.

    base_loss : keras loss function, optional

        Keras loss function. Default: :func:`keras.losses.mse`.

    """
    def __init__(self, G, base_loss=keras.losses.mse):
        if K.ndim(G) == 2:
            assert K.int_shape(G)[1] == 1, "bad shape"
            G = K.squeeze(G, axis=1)
        assert K.ndim(G) == 1, "bad shape"
        self.G = G

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
        A.set_shape([None, 1])     # we know that axis=1 must have size 1
        A = tf.squeeze(A, axis=1)  # A.shape = [batch_size]
        A = tf.cast(A, tf.int64)   # must be int (we'll use `A` for slicing)

        # check shapes
        assert K.ndim(A) == 1, "bad shape"
        assert K.ndim(Q_pred) == 2, "bad shape"
        assert K.ndim(self.G) == 1, "bad shape"
        assert K.int_shape(self.G) == K.int_shape(A), "bad shape"
        assert K.int_shape(self.G)[0] == K.int_shape(Q_pred)[0], "bad shape"

        # project onto actions taken: Q(s,.) --> Q(s,a)
        Q_pred_projected = project_onto_actions(Q_pred, A)

        # the actuall loss
        err = Q_pred_projected - self.G
        return K.mean(K.square(err))


class SemiGradientMeanSquaredErrorLoss:
    """
    Semi-gradient loss for bootstrapped MSE.

    This loss function is primarily used for updating value function by
    bootstrapping.

    For instance, in n-step bootstrapping, we minimize the bootstrapped
    MSE. The loss associated with a sampled sequence of states :math:`(S_t,
    ..., S_{t+n})` can be written as:

    .. math::

        J_t\\ =\\ \\frac12\\left(
            G^{(n)}_t + \\gamma^n\\,\\hat{v}(S_{t+n}) - \\hat{v}(S_t)\\right)^2

    The n-step return is given by:

    .. math::

        G^{(n)}_t\\ =\\
            R_{t+1}+\\gamma R_{t+2}+\\gamma^2 R_{t+3}+...
            +\\gamma^{n-1}R_{t+n}


    Parameters
    ----------
    bootstrapped_values : 1d Tensor, dtype: float, shape: [batch_size]

        These values represent the bootstrapped values produced by the
        function approximator itself (see discussion above).

        .. math::

            \\texttt{bootstrapped_values}\\ =\\ \\gamma^n\\,\\hat{v}(S_{t+n})

    base_loss : keras loss function, optional

        Keras loss function. Default: :func:`keras.losses.mse`.


    """
    def __init__(self, bootstrapped_values, base_loss=keras.losses.mse):
        self.bootstrapped_values = tf.stop_gradient(bootstrapped_values)
        self.base_loss = base_loss

    def __call__(self, Gn, y_pred):
        """
        The loss function :math:`J_t`.

        Parameters
        ----------
        Gn : 1d Tensor, dtype: float, shape: [batch_size]

            A tensor of partial returns :math:`G^{(n)}_t` (see discussion
            above).

        y_pred : 1d Tensor, dtype: float, shape: [batch_size]

            The predicted values :math:`\\hat{v}(S_t)`.

        Returns
        -------
        loss : 0d Tensor (scalar), dtype: float

            The bootstrapped MSE.

        """
        y_true = Gn + self.bootstrapped_values
        return self.base_loss(y_true, y_pred)
