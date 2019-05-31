from tensorflow import keras

from ..base.function_approximators.generic import GenericSoftmaxPolicy
from ..base.function_approximators.atari import AtariFunctionMixin
from ..utils import check_tensor


__all__ = (
    'AtariPolicy',
)


class AtariPolicy(GenericSoftmaxPolicy, AtariFunctionMixin):
    """
    A specific :term:`type-II <type-II state-action value
    function>` Q-function for Atari environments.

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

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the Adam optimizer is used,
        :class:`keras.optimizers.Adam`. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **adam_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.Adam
        <https://keras.io/optimizers/#adam>`_.

    """
    def __init__(
            self, env,
            update_strategy='ppo',
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self._init_optimizer(optimizer, adam_kwargs)
        self._init_models()
        self._check_attrs()

    def _head(self, X, variable_scope):
        layer = keras.layers.Dense(
            units=self.num_actions,
            kernel_initializer='zeros',
            name=(variable_scope + '/policy'))
        return layer(X)

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='Adv', shape=(), dtype='float')

        # computation graph
        Logits = self._forward_pass(S, variable_scope='primary')
        Logits_target = self._forward_pass(S, variable_scope='target')
        check_tensor(Logits, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(Logits_target, ndim=2, axis_size=self.num_actions, axis=1)

        # loss and target tensor (depends on self.update_strategy)
        loss, Y = self._policy_loss_and_target(Adv, Logits, Logits_target)

        # models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Y)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Logits)
        self.target_model = keras.Model(inputs=S, outputs=Logits_target)
