from tensorflow.keras import backend as K

from .base import BaseLoss, BasePolicyLoss
from ..utils import check_tensor, proba_ratio, cross_entropy, entropy


__all__ = (
    'ClippedSurrogateLoss',
    'PolicyEntropy',
    'PolicyKLDivergence',
    'VanillaPolicyLoss',
)


class VanillaPolicyLoss(BasePolicyLoss):
    """
    Plain-vanilla policy loss.

    .. math::

        L(\\theta)\\ &=\\ \\mathbb{E}_t\\left\\{
                -\\sum_ab(a|S_t)\\log\\pi_\\theta(a|S_t)\\,\\mathcal{A}(S_t, a)
            \\right\\} \\\\
            &=\\ \\mathbb{E}_t\\left\\{
                -\\log\\pi_\\theta(A_t|S_t)\\,\\mathcal{A}_t
            \\right\\}

    where :math:`\\pi_\\theta(a|s)` is the policy we're trying to learn,
    :math:`b(a|s)` is the behavior policy, and :math:`\\mathcal{A}(s, a)` is
    the advantage function, cf. Chapter 13 of `Sutton & Barto
    <http://incompleteideas.net/book/the-book-2nd.html>`_.

    This class provides a stateful implementation of a keras-compatible loss
    function that requires more input than just ``y_true`` and ``y_pred``. The
    required state that this loss function depends on is a tensor that contains
    a batch of observed advantages :math:`\\mathcal{A}_t=\\mathcal{A}(S_t,
    A_t)`.

    Parameters
    ----------
    dist_id : str

        The policy distribution id, e.g. ``'categorical'`` or ``'beta'`` for
        a softmax policy or a Beta policy, respectively.

    Adv : 1d Tensor, dtype: float, shape: [batch_size]

        The advantages, one for each time step.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    """
    def __call__(self, P, Z, sample_weight=None):
        batch_size = K.int_shape(self.Adv)[0]
        check_tensor(Z, axis_size=batch_size, axis=0)
        P.set_shape(Z.get_shape())

        # log(pi(a|s))
        logpi = -cross_entropy(P, Z, self.dist_id, Z1_is_logit=False)
        check_tensor(logpi, ndim=1, axis_size=batch_size, axis=0)

        # vanilla policy loss
        loss = -self.Adv * logpi

        # entropy H[pi]
        H = entropy(Z, self.dist_id)
        check_tensor(H, ndim=1, axis_size=batch_size, axis=0)

        # add entropy bonus (notice minus sign)
        loss = loss - self.entropy_bonus * H

        return K.mean(loss)  # aggregate of samples in batch


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
    dist_id : str

        The policy distribution id, e.g. ``'categorical'`` or ``'beta'`` for
        a softmax policy or a Beta policy, respectively.

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
    def __init__(
            self, dist_id, Adv, Z_target,
            entropy_bonus=0.01,
            epsilon=0.2):

        super().__init__(dist_id, Adv, entropy_bonus=entropy_bonus)

        self.epsilon = float(epsilon)
        self.Z_target = K.stop_gradient(Z_target)

    def __call__(self, P, Z, sample_weight=None):
        batch_size = K.int_shape(self.Adv)[0]

        # construct probability ratio, r = pi / pi_old
        r = proba_ratio(P, Z, self.Z_target, self.dist_id)
        r_clipped = K.clip(r, 1 - self.epsilon, 1 + self.epsilon)
        check_tensor(r, ndim=1, axis_size=batch_size, axis=0)
        check_tensor(r_clipped, same_as=r)

        # construct the final clipped surrogate loss (notice minus sign)
        loss = -K.minimum(r * self.Adv, r_clipped * self.Adv)

        # entropy H[pi]
        H = entropy(Z, self.dist_id)
        check_tensor(H, ndim=1, axis_size=batch_size, axis=0)

        # add entropy bonus (notice minus sign)
        loss = loss - self.entropy_bonus * H

        return K.mean(loss)


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
        self.Z_target = K.stop_gradient(Z_target)

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
        H_cross = cross_entropy(self.Z_target, Z, self.dist_id)
        H = entropy(self.Z_target, self.dist_id)
        return K.mean(H_cross - H)


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
        H = entropy(self.Z_target, self.dist_id)
        return K.mean(H)
