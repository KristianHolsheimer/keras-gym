from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import check_tensor
from ..proba_dists import CategoricalDist
from .base import BaseUpdateablePolicy


class SoftmaxPolicy(BaseUpdateablePolicy):
    """

    :term:`Updateable policy <updateable policy>` for discrete action spaces.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

    update_strategy : str, callable, optional

        The strategy for updating our policy. This determines the loss function
        that we use for our policy function approximator. If you wish to use a
        custom policy loss, you can override the
        :func:`policy_loss_with_metrics` method.

        Provided options are:

            'vanilla'
                Plain vanilla policy gradient. The corresponding (surrogate)
                loss function that we use is:

                .. math::

                    J(\\theta)\\ =\\ -\\mathcal{A}(s,a)\\,\\ln\\pi(a|s,\\theta)

            'ppo'
                `Proximal policy optimization
                <https://arxiv.org/abs/1707.06347>`_ uses a clipped proximal
                loss:

                .. math::

                    J(\\theta)\\ =\\ \\min\\Big(
                        r(\\theta)\\,\\mathcal{A}(s,a)\\,,\\
                        \\text{clip}\\big(
                            r(\\theta), 1-\\epsilon, 1+\\epsilon\\big)
                                \\,\\mathcal{A}(s,a)\\Big)

                where :math:`r(\\theta)` is the probability ratio:

                .. math::

                    r(\\theta)\\ =\\ \\frac
                        {\\pi(a|s,\\theta)}
                        {\\pi(a|s,\\theta_\\text{old})}

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

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __call__(self, s, use_target_model=False):
        # Because we want our categorical samples to be differentiable,
        # self.predict_model cannot return fully deterministic samples.
        # We therefore perform the final sampling in the numpy layer.
        p = super().__call__(s, use_target_model)  # p is almost deterministic
        a = self.random.choice(self.num_actions, p=p)
        return a

    def _init_models(self):
        S = keras.Input(
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype, name='policy/S')
        A = keras.Input(
            shape=[self.num_actions], dtype='float', name='policy/A')
        Adv = keras.Input(shape=(), dtype='float', name='policy/Adv')

        # forward pass
        X = self.function_approximator.body(S)
        logits = self.function_approximator.head_pi(X)
        check_tensor(logits, ndim=2, axis_size=self.num_actions, axis=1)

        # apply available-action mask (optional)
        if hasattr(self, 'available_actions_mask'):
            check_tensor(self.available_actions_mask, ndim=2, dtype='bool')
            # set logits to large negative values for unavailable actions
            logits = keras.layers.Lambda(
                lambda x: K.switch(
                    self._available_actions, x, -1e3 * K.ones_like(x)),
                name=('policy/logits/masked'))(logits)

        # special layers
        A_sample = keras.layers.Lambda(
            lambda x: CategoricalDist(logits=x).sample())(logits)
        A_greedy = keras.layers.Lambda(K.argmax)(logits)

        # output models
        self.predict_model = keras.Model(S, A_sample)
        self.target_model = keras.models.clone_model(self.predict_model)
        self.predict_greedy_model = keras.Model(S, A_greedy)
        self.target_greedy_model = keras.models.clone_model(
            self.predict_greedy_model)
        self.predict_param_model = keras.Model(S, logits)
        self.target_param_model = keras.models.clone_model(
            self.predict_param_model)

        # loss and target tensor
        self.dist = CategoricalDist(logits=logits)
        self.target_dist = CategoricalDist(logits=self.target_param_model(S))
        loss, metrics = self.policy_loss_with_metrics(Adv, A)

        # models
        self.train_model = keras.Model([S, A, Adv], logits)
        self.train_model.add_loss(loss)
        for name, metric in metrics.items():
            self.train_model.add_metric(metric, name=name, aggregation='mean')
        self.train_model.compile(
            optimizer=self.function_approximator.optimizer)
