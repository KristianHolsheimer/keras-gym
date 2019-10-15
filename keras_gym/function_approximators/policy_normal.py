from tensorflow import keras

from ..utils import check_tensor
from ..proba_dists import NormalDist
from .base import BaseUpdateablePolicy


class GaussianPolicy(BaseUpdateablePolicy):
    """

    An :term:`updateable policy` for environments with a continuous action
    space, i.e. a :class:`Box <gym.spaces.Box>`. It models the policy
    :math:`\\pi_\\theta(a|s)` as a normal distribution with conditional
    parameters :math:`(\\mu_\\theta(s), \\sigma_\\theta(s))`.

    .. important::

        This environment requires that the ``env`` is with:

        .. code::

            env = km.wrappers.BoxToReals(env)

        This wrapper decompactifies the Box action space.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

    update_strategy : str, optional

        The strategy for updating our policy. This typically determines the
        loss function that we use for our policy function approximator.

        Options are:

            'vanilla'
                Plain vanilla policy gradient. The corresponding (surrogate)
                loss function that we use is:

                .. math::

                    J(\\theta)\\ =\\ \\hat{\\mathbb{E}}_t
                        \\left\\{
                            -\\mathcal{A}_t\\,\\log\\pi_\\theta(A_t|S_t)
                        \\right\\}

                where :math:`\\mathcal{A}_t=\\mathcal{A}(S_t,A_t)` is the
                advantage at time step :math:`t`.

            'ppo'
                `Proximal policy optimization
                <https://arxiv.org/abs/1707.06347>`_ uses a clipped proximal
                loss:

                .. math::

                    J(\\theta)\\ =\\ \\hat{\\mathbb{E}}_t
                        \\left\\{
                            \\min\\Big(
                                \\rho_t(\\theta)\\,\\mathcal{A}_t\\,,\\
                                \\tilde{\\rho}_t(\\theta)\\,\\mathcal{A}_t
                            \\Big)
                        \\right\\}

                where :math:`\\rho_t(\\theta)` is the probability ratio:

                .. math::

                    \\rho_t(\\theta)\\ =\\ \\frac
                        {\\pi_\\theta(A_t|S_t)}
                        {\\pi_{\\theta_\\text{old}}(A_t|S_t)}

                and :math:`\\tilde{\\rho}_t(\\theta)` is its clipped version:

                .. math::

                    \\tilde{\\rho}_t(\\theta)\\ =\\ \\text{clip}\\big(
                            \\rho_t(\\theta), 1-\\epsilon, 1+\\epsilon\\big)

            'cross_entropy'
                Straightforward categorical cross-entropy (from logits). This
                loss function does *not* make use of the advantages
                :term:`Adv`. Instead, it minimizes the cross entropy between
                the behavior policy :math:`\\pi_b(a|s)` and the learned policy
                :math:`\\pi_\\theta(a|s)`:

                .. math::

                    J(\\theta)\\ =\\ \\hat{\\mathbb{E}}_t\\left\\{
                        -\\sum_a \\pi_b(a|S_t)\\, \\log \\pi_\\theta(a|S_t)
                    \\right\\}

    ppo_clip_eps : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_beta : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    """

    def _init_models(self):
        S = keras.Input(
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype, name='policy/S')
        A = keras.Input(
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype, name='policy/A')
        Adv = keras.Input(shape=(), dtype='float', name='policy/Adv')

        # forward pass
        X = self.function_approximator.body(S)
        mu, logvar = self.function_approximator.head_pi(X)
        check_tensor(mu, ndim=2, axis_size=self.actions_ndim, axis=1)
        check_tensor(logvar, same_as=mu)

        # special layers
        A_sample = keras.layers.Lambda(
            lambda args: NormalDist(*args).sample())([mu, logvar])
        A_greedy = mu

        # output models
        self.predict_model = keras.Model(S, A_sample)
        self.target_model = keras.models.clone_model(self.predict_model)
        self.predict_greedy_model = keras.Model(S, A_greedy)
        self.target_greedy_model = keras.models.clone_model(
            self.predict_greedy_model)
        self.predict_param_model = keras.Model(S, [mu, logvar])
        self.target_param_model = keras.models.clone_model(
            self.predict_param_model)

        # loss and target tensor (depends on self.update_strategy)
        self.dist = NormalDist(mu=mu, logvar=logvar)
        self.target_dist = NormalDist(*self.target_param_model(S))
        loss, metrics = self.policy_loss_with_metrics(Adv, A)

        # models
        self.train_model = keras.Model([S, A, Adv], [mu, logvar])
        self.train_model.add_loss(loss)
        for name, metric in metrics.items():
            self.train_model.add_metric(metric, name=name, aggregation='mean')
        self.train_model.compile(
            optimizer=self.function_approximator.optimizer)
