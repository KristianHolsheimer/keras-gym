from tensorflow import keras

from ..losses import ProjectedSemiGradientLoss, Huber
from ..base.function_approximators.generic import GenericV, GenericQTypeII
from ..base.function_approximators.atari import AtariFunctionMixin

__all__ = (
    'AtariV',
    'AtariQ',
)


class AtariV(GenericV, AtariFunctionMixin):
    """
    A specific :term:`state value function` for Atari environments.

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

    bootstrap_with_target_model : bool, optional

        Whether to use the :term:`target_model` when constructing a
        bootstrapped target. If False (default), the primary
        :term:`predict_model` is used.

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
            gamma=0.99,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self._init_optimizer(optimizer, adam_kwargs)
        self._init_models()
        self._check_attrs()

    def _head(self, X, variable_scope):
        layer = keras.layers.Dense(
            units=1,
            kernel_initializer='zeros',
            name=(variable_scope + '/V'))
        return layer(X)

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)

        # regular computation graph
        V = self._forward_pass(S, variable_scope='primary')

        # regular models
        self.train_model = keras.Model(inputs=S, outputs=V)
        self.train_model.compile(
            loss=Huber(), optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=V)

        # target model
        V_target = self._forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=V_target)


class AtariQ(GenericQTypeII, AtariFunctionMixin):
    """
    A specific :term:`type-II <type-II state-action value
    function>` Q-function for Atari environments.

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

    bootstrap_with_target_model : bool, optional

        Whether to use the :term:`target_model` when constructing a
        bootstrapped target. If False (default), the primary
        :term:`predict_model` is used.

    update_strategy : str, optional

        The update strategy that we use to select the (would-be) next-action
        :math:`A_{t+n}` in the bootsrapped target:

        .. math::

            G^{(n)}_t\\ =\\ R^{(n)}_t + \\gamma^n Q(S_{t+n}, A_{t+n})

        Options are:

            'sarsa'
                Sample the next action, i.e. use the action that was actually
                taken.

            'q_learning'
                Take the action with highest Q-value under the current
                estimate, i.e. :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n}, a)`.
                This is an off-policy method.

            'double_q_learning'
                Same as 'q_learning', :math:`A_{t+n} = \\arg\\max_aQ(S_{t+n},
                a)`, except that the value itself is computed using the
                :term:`target_model` rather than the primary model, i.e.

                .. math::

                    A_{t+n}\\ &=\\
                        \\arg\\max_aQ_\\text{primary}(S_{t+n}, a)\\\\
                    G^{(n)}_t\\ &=\\ R^{(n)}_t
                        + \\gamma^n Q_\\text{target}(S_{t+n}, A_{t+n})

            'expected_sarsa'
                Similar to SARSA in that it's on-policy, except that we take
                the expectated Q-value rather than a sample of it, i.e.

                .. math::

                    G^{(n)}_t\\ =\\ R^{(n)}_t
                        + \\gamma^n\\sum_a\\pi(a|s)\\,Q(S_{t+n}, a)

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
            gamma=0.99,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='q_learning',
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
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
            name=(variable_scope + '/Q'))
        return layer(X)

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)
        G = keras.Input(name='value/G', shape=(), dtype='float')

        # regular computation graph
        Q = self._forward_pass(S, variable_scope='primary')

        # loss
        loss = ProjectedSemiGradientLoss(G, base_loss=Huber())

        # regular models
        self.train_model = keras.Model(inputs=[S, G], outputs=Q)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Q)

        # target model
        Q_target = self._forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=Q_target)
