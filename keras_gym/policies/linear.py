from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import check_tensor
from ..base.function_approximators.linear import LinearFunctionMixin
from ..base.function_approximators.generic import GenericSoftmaxPolicy


__all__ = (
    'LinearSoftmaxPolicy',
)


class LinearSoftmaxPolicy(GenericSoftmaxPolicy, LinearFunctionMixin):
    """
    Linear-model implementation for :term:`updateable policies <updateable
    policy>` for discrete action spaces.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    update_strategy : str, optional

        The strategy for updating our policy. This typically determines the
        loss function that we use for our policy function approximator.

        Options are:

            'vanilla'
                Plain vanilla policy gradient. The corresponding (surrogate)
                objective that we use is:

                .. math::

                    J(\\theta)\\ =\\ -\\mathcal{A}(s,a)\\,\\ln\\pi(a|s,\\theta)

            'ppo'
                `Proximal policy optimization
                <https://arxiv.org/abs/1707.06347>`_ uses a clipped surrogate
                objective:

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

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            'full_quadratic'
                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be the batch
                size and number of features. Then, the input shape is
                :math:`(b, n)` and the output shape is :math:`(b, (n + 1) (n
                + 2) / 2 - 1))`.

                **Note:** This option requires the `tensorflow` backend.

            'elementwise_quadratic'
                This option generates element-wise quadratic interactions,
                which only include linear and quadratic terms. It does *not*
                include bilinear terms or an intercept. Let :math:`b` and
                :math:`n` be the batch size and number of features. Then, the
                input shape is :math:`(b, n)` and the output shape is
                :math:`(b, 2n)`.

        Otherwise, a custom interaction layer can be passed as well. If left
        unspecified (`interaction=None`), the interaction layer is omitted
        altogether.

    ppo_clipping : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the plain vanilla `SGD
        <https://keras.io/optimizers/#sgd>`_ optimizer is used. See `keras
        documentation <https://keras.io/optimizers/>`_ for other options.

    **sgd_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.SGD
        <https://keras.io/optimizers/#sgd>`_.

    """
    def __init__(
            self, env,
            update_strategy='vanilla',
            interaction=None,
            ppo_clipping=0.2,
            entropy_bonus=0.01,
            random_seed=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            update_strategy=update_strategy,
            ppo_clipping=ppo_clipping,
            entropy_bonus=entropy_bonus,
            random_seed=random_seed,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.interaction = interaction
        self._init_interaction_layer(interaction)
        self._init_optimizer(optimizer, sgd_kwargs)
        self._init_models(output_dim=self.num_actions)
        self._check_attrs()

    def _init_models(self, output_dim):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='policy/S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='policy/Adv', shape=(), dtype='float')

        def forward_pass(S, variable_scope):
            def v(name):
                return '{}/{}'.format(variable_scope, name)

            if K.ndim(S) > 2:
                S = keras.layers.Flatten(S)

            if self.interaction_layer is not None:
                S = self.interaction_layer(S)

            dense_layer = keras.layers.Dense(
                output_dim, kernel_initializer='zeros', name=v('weights'))

            return dense_layer(S)

        # computation graph
        Z = forward_pass(S, variable_scope='primary')
        Z_target = forward_pass(S, variable_scope='target')
        check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(Z_target, ndim=2, axis_size=self.num_actions, axis=1)

        # loss and target tensor (depends on self.update_strategy)
        loss = self._policy_loss(Adv, Z_target)

        # models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Z)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Z)
        self.target_model = keras.Model(inputs=S, outputs=Z_target)
