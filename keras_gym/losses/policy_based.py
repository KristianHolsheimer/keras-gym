import tensorflow as tf
from tensorflow.keras import backend as K

from .base import BaseLoss, BasePolicyLoss
from ..utils import project_onto_actions_tf, check_tensor, log_softmax_tf


__all__ = (
    'SoftmaxPolicyLossWithLogits',
    'ClippedSurrogateLoss',
    'PolicyEntropy',
    'PolicyKLDivergence',
)


class SoftmaxPolicyLossWithLogits(BasePolicyLoss):
    """
    Softmax-policy loss (with logits).

    This class provides a stateful implementation of a keras-compatible loss
    function that requires more input than just ``y_true`` and ``y_pred``. The
    required state that this loss function depends on is a batch of so-called
    *advantages* :math:`\\mathcal{A}(s, a)`, which are essentially shifted
    returns, cf. Chapter 13 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_. The advantage
    function is often defined as :math:`\\mathcal{A}(s, a) = q(s, a) - v(s)`.
    The baseline function :math:`v(s)` can be anything you like; a common
    choice is :math:`v(s) = \\sum_a\\pi(a|s)\\,q(s,a)`, in which case
    :math:`\\mathcal{A}(s, a)` is a proper advantage function with vanishing
    expectation value.

    This loss function is actually a surrogate loss function defined in such a
    way that its gradient is the same as what one would get by taking a true
    policy gradient.

    Parameters
    ----------

    Adv : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one for each time step.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    """
    @staticmethod
    def logpi_surrogate(Z):
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
        Z : 2d Tensor, shape: [batch_size, num_actions]

            The predicted logits of the softmax policy, a.k.a. ``y_pred``.

        Returns
        -------
        logpi_surrogate : Tensor, same shape as input

            The surrogate for :math:`\\log\\pi_\\theta(a|s)`.

        """
        check_tensor(Z, ndim=2)
        pi = K.stop_gradient(K.softmax(Z, axis=1))
        Z_mean = K.expand_dims(tf.einsum('ij,ij->i', pi, Z), axis=1)
        return Z - Z_mean

    def __call__(self, P, Z, sample_weight=None):
        batch_size = K.int_shape(self.Adv)[0]

        # check shapes
        check_tensor(P, ndim=2, axis_size=batch_size, axis=0)
        check_tensor(Z, ndim=2, axis_size=batch_size, axis=0)

        # construct the surrogate for logpi(.|s)
        logpi_all = self.logpi_surrogate(Z)  # [batch_size, num_actions]

        # project onto actions taken: logpi(.|s) --> logpi(a|s)
        logpi = tf.einsum('ij,ij->i', logpi_all, P)  # shape: [batch_size]

        # construct the final surrogate loss
        surrogate_loss = -K.mean(self.Adv * logpi)

        # entropy bonus term (notice minus sign)
        L_entropy = -self.entropy_bonus * PolicyEntropy()(P, Z)

        return surrogate_loss + L_entropy


class ClippedSurrogateLoss(BasePolicyLoss):
    """

    The clipped surrogate loss used in `PPO
    <https://arxiv.org/abs/1707.06347>`_.

    .. math::

        L(\\theta)\\ =\\ -\\hat{\\mathbb{E}}_t\\min\\big(
            r_t(\\theta)          \\,\\mathcal{A}_t\\,,\\
            r^\\epsilon_t(\\theta)\\,\\mathcal{A}_t\\big)

    where :math:`\\mathcal{A}_t=\\mathcal{A}(S_t,A_t)` is the (sampled)
    advantage and :math:`r_t(\\theta)` is the probability ratio:

    .. math::

        r_t(\\theta)\\ =\\ \\frac
            {\\pi(A_t|S_t,\\theta)}
            {\\pi(A_t|S_t,\\theta_\\text{old})}

    Also, :math:`r^\\epsilon_t(\\theta)` is the *clipped* probability ratio:

    .. math::

        r^\\epsilon_t(\\theta)\\ =\\ \\texttt{clip}\\big(
            r_t(\\theta), 1-\\epsilon, 1+\\epsilon\\big)

    Parameters
    ----------

    Adv : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one for each time step.

    Z_target : 2d Tensor, shape: [batch_size, num_actions]

        The predicted logits of the :term:`target_model` of the policy object.
        In policy-gradient methods, the :term:`target_model` is effectively the
        *behavior* policy, i.e. the one that actually generates the
        observations.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    epsilon : float between 0 and 1, optional

        Hyperparameter that determines how we clip the surrogate loss.

    """
    def __init__(self, Adv, Z_target, entropy_bonus=0.01, epsilon=0.2):
        super().__init__(Adv, entropy_bonus=entropy_bonus)
        self.epsilon = float(epsilon)

        check_tensor(Z_target, ndim=2)
        self.logpi_old = K.stop_gradient(log_softmax_tf(Z_target, axis=1))

    def __call__(self, P, Z, sample_weight=None):
        batch_size = K.int_shape(self.Adv)[0]

        # check shapes
        check_tensor(P, ndim=2, axis_size=batch_size, axis=0)
        check_tensor(Z, ndim=2, axis_size=batch_size, axis=0)

        # construct probability ratio, r = pi / pi_old
        logpi = log_softmax_tf(Z)
        r = K.exp(logpi - self.logpi_old)  # shape: [batch_size, num_actions]
        r = tf.einsum('ij,ij->i', r, P)    # shape: [batch_size]

        # construct the final clipped surrogate loss (notice minus sign)
        L_clip = -K.mean(K.minimum(
            r * self.Adv,
            K.clip(r, 1 - self.epsilon, 1 + self.epsilon) * self.Adv))

        # entropy bonus term (notice minus sign)
        L_entropy = -self.entropy_bonus * PolicyEntropy()(None, Z)

        return L_clip + L_entropy


class PolicyKLDivergence(BaseLoss):
    """

    Computes the KL divergence between the current policy and the old
    (behavior) policy:

    .. math::

        \\hat{\\mathbb{E}}_t\\left\\{\\,
            KL[\\pi_{\\theta_\\text{old}}(.|S_t), \\pi_\\theta(.|S_t)]
        \\right\\}
        \\ =\\ \\hat{\\mathbb{E}}_t\\left\\{
            \\sum_a \\pi_{\\theta_\\text{old}}(a|S_t)\\,
                \\log\\frac{\\pi_{\\theta_\\text{old}}(a|S_t)}
                           {\\pi_\\theta(a|S_t)}\\right\\}


    Parameters
    ----------

    Z_target : 2d Tensor, shape: [batch_size, num_actions]

        The predicted logits of the :term:`target_model` of the policy object.
        In policy-gradient methods, the :term:`target_model` is effectively the
        *behavior* policy, i.e. the one that actually generates the
        observations.

    """
    def __init__(self, Z_target):
        check_tensor(Z_target, ndim=2)
        self.logpi_old = K.stop_gradient(log_softmax_tf(Z_target, axis=1))
        self.pi_old = K.stop_gradient(K.softmax(Z_target, axis=1))

    def __call__(self, P, Z, sample_weight=None):
        """

        Compute the the old-vs-new policy KL divergence.

        Parameters
        ----------
        P : Tensor

            This input is ignored.

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
        check_tensor(Z, ndim=2)
        logpi = log_softmax_tf(Z)
        kl_div = tf.einsum('ij,ij->i', self.pi_old, self.logpi_old - logpi)
        return K.mean(kl_div)


class PolicyEntropy(BaseLoss):
    """

    Computes the entropy of a policy:

    .. math::

        \\hat{\\mathbb{E}}_t\\left\\{S[\\pi(.|S_t)]\\right\\}
        \\ =\\ \\hat{\\mathbb{E}}_t\\left\\{
            -\\sum_a \\pi(a|S_t)\\,\\log\\pi(a|S_t)\\right\\}

    """
    def __init__(self):
        pass

    def __call__(self, A, Z, sample_weight=None):
        """
        Compute the action-space entropy of a policy.

        Parameters
        ----------
        P : Tensor

            This input is ignored.

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
        check_tensor(Z, ndim=2)
        logpi = log_softmax_tf(Z)
        pi = K.exp(logpi)
        entropy = -tf.einsum('ij,ij->i', pi, logpi)
        return K.mean(entropy)
