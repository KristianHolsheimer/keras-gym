from tensorflow import keras

from ..base.function_approximators.generic import GenericSoftmaxPolicy
from ..base.function_approximators.atari import AtariFunctionMixin
from ..value_functions import GenericV, AtariV
from ..utils import check_tensor
from ..losses import Huber
from .actor_critic import ActorCritic


__all__ = (
    'AtariPolicy',
    'AtariActorCritic',
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

    ppo_clipping : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

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
            ppo_clipping=0.2,
            entropy_bonus=0.01,
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            update_strategy=update_strategy,
            ppo_clipping=ppo_clipping,
            entropy_bonus=entropy_bonus,
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

        S = keras.Input(name='policy/S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='policy/Adv', shape=(), dtype='float')

        # computation graph
        Z = self._forward_pass(S, variable_scope='primary')
        Z_target = self._forward_pass(S, variable_scope='target')
        check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(Z_target, ndim=2, axis_size=self.num_actions, axis=1)

        # loss and target tensor (depends on self.update_strategy)
        loss = self._policy_loss(Adv, Z_target)

        # models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Z)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Z)
        self.target_model = keras.Model(inputs=S, outputs=Z_target)


class AtariActorCritic(ActorCritic, AtariFunctionMixin):
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

    policy_update_strategy : str, optional

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
            gamma=0.99,
            bootstrap_n=1,
            update_strategy='ppo',
            ppo_clipping=0.2,
            entropy_bonus=0.01,
            optimizer=None,
            **adam_kwargs):

        self.policy = GenericSoftmaxPolicy(
            env=env,
            update_strategy=update_strategy,
            ppo_clipping=ppo_clipping,
            entropy_bonus=entropy_bonus,
            random_seed=None,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.value_function = GenericV(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=True,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self._init_optimizer(optimizer, adam_kwargs)
        self._init_models()
        self._check_attrs()

    def _head(self):
        raise NotImplementedError

    def _policy_head(self, X, variable_scope):
        return AtariPolicy._head(self, X, variable_scope)

    def _value_head(self, X, variable_scope):
        return AtariV._head(self, X, variable_scope)

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='actor_critic/S', shape=shape, dtype=dtype)
        G = keras.Input(name='actor_critic/G', shape=[1], dtype='float')

        # shared part of the computation graph
        X = self._shared_forward_pass(S, variable_scope='primary')
        X_target = self._shared_forward_pass(S, variable_scope='target')

        # value head
        V = self._value_head(X, variable_scope='primary')
        V_target = self._value_head(X_target, variable_scope='target')

        # policy head (Z == logits)
        Z = self._policy_head(X, variable_scope='primary')
        Z_target = self._policy_head(X_target, variable_scope='target')

        # consistency checks
        check_tensor(V, ndim=2, axis_size=1, axis=1)
        check_tensor(V_target, ndim=2, axis_size=1, axis=1)
        check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(Z_target, ndim=2, axis_size=self.num_actions, axis=1)

        # policy models
        self.policy.predict_model = keras.Model(inputs=S, outputs=Z)
        self.policy.target_model = keras.Model(inputs=S, outputs=Z_target)

        # value models
        self.value_function.predict_model = keras.Model(inputs=S, outputs=V)
        self.value_function.target_model = keras.Model(
            inputs=S, outputs=V_target)

        # loss and target tensor (depends on self.update_strategy)
        Adv = G - V_target
        policy_loss = self.policy._policy_loss(Adv, Z_target)
        value_loss = Huber()

        # joint train model
        self.train_model = keras.Model(inputs=[S, G], outputs=[Z, V])
        self.train_model.compile(
            loss=[policy_loss, value_loss], optimizer=self.optimizer)
        self.train_model.summary()

    def _check_attrs(self):
        pass
