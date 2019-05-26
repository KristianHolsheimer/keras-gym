from tensorflow import keras
from tensorflow.keras import backend as K

from ..losses import SoftmaxPolicyLossWithLogits
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

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

    update_strategy : str, optional

        The strategy for updating our policy. This typically determines the
        loss function that we use for our policy function approximator.

        Options are:

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

                #TODO: to be implemented -Kris

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

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the plain vanilla SGD
        optimizer is used, :class:`keras.optimizers.SGD`. See `keras
        documentation <https://keras.io/optimizers/>`_ for more details.

    **sgd_kwargs : keyword arguments

        Keyword arguments for :class:`keras.optimizers.SGD`. See `keras docs
        <https://keras.io/optimizers/#sgd>`_ for more details.

    """
    def __init__(
            self, env,
            update_strategy='vanilla',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            update_strategy=update_strategy,
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

        S = keras.Input(name='S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='Adv', shape=(), dtype='float')

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
        Logits = forward_pass(S, variable_scope='primary')

        # loss
        if self.update_strategy == 'vanilla':
            loss = SoftmaxPolicyLossWithLogits(Adv)
        elif self.update_strategy == 'trpo':
            raise NotImplementedError("update_strategy == 'trpo'")  # TODO
        elif self.update_strategy == 'ppo':
            raise NotImplementedError("update_strategy == 'ppo'")  # TODO
        else:
            raise ValueError(
                "unknown update_strategy '{}'".format(self.update_strategy))

        # regular models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Logits)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Logits)

        # optional models
        self.target_model = None
