from abc import abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import check_tensor
from ..base.mixins import NumActionsMixin
from ..losses import Huber, ProjectedSemiGradientLoss
from .base import BaseV, BaseQTypeI, BaseQTypeII, BaseSoftmaxPolicy
from .actor_critic import ActorCritic


__all__ = (
    'FunctionApproximator',
    'V',
    'QTypeI',
    'QTypeII',
    'SoftmaxPolicy',
    'ConjointActorCritic',
)


class FunctionApproximator(NumActionsMixin):
    """
    A generic function approximator.

    This is the central object object that provides an interface between a
    gym-type environment and function approximators like :term:`value functions
    <state value function>` and :term:`updateable policies <updateable
    policy>`.

    In order to create a valid function approximator, you need to implement the
    :term:`body` method. For example, to implement a simple multi-layer
    perceptron function approximator you would do something like:

    .. code:: python

        import gym
        import keras_gym as km
        from tensorflow.keras.layers import Flatten, Dense

        class MLP(km.FunctionApproximator):
            \"\"\" multi-layer perceptron with one hidden layer \"\"\"
            def body(self, S, variable_scope):
                X = Flatten()(S)
                X = Dense(units=4, name=(variable_scope + '/hidden'))(X)
                return X

        # environment
        env = gym.make(...)

        # generic function approximator
        mlp = MLP(env, lr=0.001)

        # policy and value function
        pi, v = km.SoftmaxPolicy(mlp), km.V(mlp)

    The default :term:`heads <head>` are simple (multi) linear regression
    layers, which can be overridden by your own implementation.

    Parameters
    ----------
    env : environment

        A gym-style environment.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the function approximator's
        DEFAULT_OPTIMIZER is used. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **optimizer_kwargs : keyword arguments

        Keyword arguments for the optimizer. This is useful when you want to
        use the default optimizer with a different setting, e.g. changing the
        learning rate.

    """
    DEFAULT_OPTIMIZER = keras.optimizers.Adam
    VALUE_LOSS_FUNCTION = Huber()

    def __init__(self, env, optimizer=None, **optimizer_kwargs):
        self.env = env
        self._init_optimizer(optimizer, optimizer_kwargs)

    def head_v(self, X, variable_scope):
        """
        This is the :term:`state value <state value function>` head. It returns
        a scalar V-value :math:`v(s)\\in\\mathbb{R}`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Q_s : nd Tensor, shape: [batch_size, num_actions]

            The output :term:`state values <V>` :math:`v(s)\\in\\mathbb{R}`.

        """
        linear_regression_layer = keras.layers.Dense(
            units=1,
            kernel_initializer='zeros',
            name=(variable_scope + '/value'))
        return linear_regression_layer(X)

    def head_q1(self, X, variable_scope):
        """
        This is the :term:`type-I <type-I state-action value function>`
        Q-value head. It returns a scalar Q-value
        :math:`q(s,a)\\in\\mathbb{R}`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Q_sa : nd Tensor, shape: [batch_size, num_actions]

            The output :term:`type-I <Q_sa>` Q-values
            :math:`q(s,a)\\in\\mathbb{R}`.

        """
        linear_regression_layer = keras.layers.Dense(
            units=1,
            kernel_initializer='zeros',
            name=(variable_scope + '/qtype1'))
        return linear_regression_layer(X)

    def head_q2(self, X, variable_scope):
        """
        This is the :term:`type-II <type-II state-action value function>`
        Q-value head. It returns a vector of Q-values
        :math:`q(s,.)\\in\\mathbb{R}^n`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Q_s : nd Tensor, shape: [batch_size, num_actions]

            The output :term:`type-II <Q_s>` Q-values
            :math:`q(s,.)\\in\\mathbb{R}^n`.

        """
        multilinear_regression_layer = keras.layers.Dense(
            units=self.num_actions,
            kernel_initializer='zeros',
            name=(variable_scope + '/qtype2'))
        return multilinear_regression_layer(X)

    def head_pi(self, X, variable_scope):
        """
        This is the policy head. It returns logits, i.e. not probabilities. Use
        a softmax to turn the output into probabilities.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Z : nd Tensor, shape: [batch_size, num_actions]

            The output :term:`logits <Z>` :math:`z\\in\\mathbb{R}^n`, from
            which we can compute a vector of action probabilities:

            .. math::

                \\pi(.|s)\\ =\\ \\text{softmax}(z)

        """
        multilinear_regression_layer = keras.layers.Dense(
            units=self.num_actions,
            kernel_initializer='zeros',
            name=(variable_scope + '/policy'))
        return multilinear_regression_layer(X)

    @abstractmethod
    def body(self, S, variable_scope):
        """
        This is the part of the computation graph that may be shared between
        e.g. policy (actor) and value function (critic). It is typically the
        part of a neural net that does most of the heavy lifting. One may think
        of the :func:`body` as an elaborate automatic feature extractor.

        Parameters
        ----------
        S : nd Tensor: shape: [batch_size, ...]

            The input state observation.

        variable_scope : str {'primary', 'target'}

            The variable scope is a string that specifies whether the
            body computation graph is part of the **primary** or **target**
            network. See also :term:`target_model`.

        """
        pass

    def _init_optimizer(self, optimizer, optimizer_kwargs):
        if optimizer is None:
            self.optimizer = self.DEFAULT_OPTIMIZER(**optimizer_kwargs)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "unknown optimizer, expected a keras.optimizers.Optimizer or "
                "None (which sets the default keras.optimizers.Adam "
                "optimizer)")


class V(BaseV):
    """
    A :term:`state value function` :math:`s\\mapsto v(s)`.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

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

    """
    def __init__(
            self, function_approximator,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False):

        BaseV.__init__(
            self,
            env=function_approximator.env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.function_approximator = function_approximator
        self._init_models()
        self._check_attrs()

    def forward_pass(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')
        X = self.function_approximator.body(S, variable_scope)
        V = self.function_approximator.head_v(X, variable_scope)
        return V

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)

        # regular computation graph
        V = self.forward_pass(S, variable_scope='primary')

        # regular models
        self.train_model = keras.Model(inputs=S, outputs=V)
        self.train_model.compile(
            loss=self.function_approximator.VALUE_LOSS_FUNCTION,
            optimizer=self.function_approximator.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=V)

        # target model
        V_target = self.forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=V_target)


class QTypeI(BaseQTypeI):
    """
    A :term:`type-I state-action value function` :math:`(s,a)\\mapsto q(s,a)`.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

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
        :math:`A_{t+n}` in the bootstrapped target:

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

    """
    def __init__(
            self, function_approximator,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='q_learning'):

        super().__init__(
            env=function_approximator.env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.function_approximator = function_approximator
        self._init_models()
        self._check_attrs()

    def forward_pass(self, S, A, variable_scope):
        assert variable_scope in ('primary', 'target')

        def kronecker_product(args):
            S, A = args
            A = K.one_hot(A, self.num_actions)
            return tf.einsum('ij,ik->ijk', S, A)

        # first combine inputs
        S = keras.layers.Flatten()(S) if K.ndim(S) > 2 else S
        X = keras.layers.Lambda(kronecker_product)([S, A])

        X = self.function_approximator.body(X, variable_scope)
        Q = self.function_approximator.head_q1(X, variable_scope)
        return Q

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)
        A = keras.Input(name='value/A', shape=(), dtype='int32')

        # regular models
        Q = self.forward_pass(S, A, variable_scope='primary')
        self.train_model = keras.Model(inputs=[S, A], outputs=Q)
        self.train_model.compile(
            loss=self.function_approximator.VALUE_LOSS_FUNCTION,
            optimizer=self.function_approximator.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for type-I

        # target model
        Q_target = self.forward_pass(S, A, variable_scope='target')
        self.target_model = keras.Model([S, A], Q_target)


class QTypeII(BaseQTypeII):
    """
    A :term:`type-II state-action value function` :math:`s\\mapsto q(s,.)`.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

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
        :math:`A_{t+n}` in the bootstrapped target:

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

    """
    def __init__(
            self, function_approximator,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='q_learning'):

        super().__init__(
            env=function_approximator.env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.function_approximator = function_approximator
        self._init_models()
        self._check_attrs()

    def forward_pass(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')
        X = self.function_approximator.body(S, variable_scope)
        Q = self.function_approximator.head_q2(X, variable_scope)
        return Q

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)
        G = keras.Input(name='value/G', shape=(), dtype='float')

        # regular computation graph
        Q = self.forward_pass(S, variable_scope='primary')

        # loss
        loss = ProjectedSemiGradientLoss(
            G, base_loss=self.function_approximator.VALUE_LOSS_FUNCTION)

        # regular models
        self.train_model = keras.Model(inputs=[S, G], outputs=Q)
        self.train_model.compile(
            loss=loss, optimizer=self.function_approximator.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Q)

        # target model
        Q_target = self.forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=Q_target)


class SoftmaxPolicy(BaseSoftmaxPolicy):
    """

    An :term:`updateable policy` for environments with a discrete action space.

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

    ppo_clipping : float, optional

        The clipping parameter :math:`\\epsilon` in the PPO clipped surrogate
        loss. This option is only applicable if ``update_strategy='ppo'``.

    entropy_bonus : float, optional

        The coefficient of the entropy bonus term in the policy objective.

    """
    def __init__(
            self, function_approximator,
            update_strategy='ppo',
            ppo_clipping=0.2,
            entropy_bonus=0.01):

        super().__init__(
            env=function_approximator.env,
            update_strategy=update_strategy,
            ppo_clipping=ppo_clipping,
            entropy_bonus=entropy_bonus,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.function_approximator = function_approximator
        self._init_models()
        self._check_attrs()

    def forward_pass(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')
        X = self.function_approximator.body(S, variable_scope)
        Z = self.function_approximator.head_pi(X, variable_scope)

        if hasattr(self, 'available_actions_mask'):
            check_tensor(self.available_actions_mask, ndim=2, dtype='bool')
            # set logits to large negative values for unavailable actions
            Z = keras.layers.Lambda(
                lambda Z: K.switch(
                    self._available_actions, Z, -1e3 * K.ones_like(Z)),
                name=(variable_scope + '/policy/masked'))(Z)

        return Z

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='policy/S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='policy/Adv', shape=(), dtype='float')

        # computation graph
        Z = self.forward_pass(S, variable_scope='primary')
        Z_target = self.forward_pass(S, variable_scope='target')
        check_tensor(Z, ndim=2, axis_size=self.num_actions, axis=1)
        check_tensor(Z_target, ndim=2, axis_size=self.num_actions, axis=1)

        # loss and target tensor (depends on self.update_strategy)
        loss = self._policy_loss(Adv, Z_target)

        # models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Z)
        self.train_model.compile(
            loss=loss, optimizer=self.function_approximator.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Z)
        self.target_model = keras.Model(inputs=S, outputs=Z_target)


class ConjointActorCritic(ActorCritic):
    """

    An :term:`actor-critic` whose :term:`body` is shared between actor and
    critic.

    Parameters
    ----------
    function_approximator : FunctionApproximator object

        The main :term:`function approximator`.

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

    """
    def __init__(
            self, function_approximator,
            gamma=0.99,
            bootstrap_n=1,
            update_strategy='ppo',
            ppo_clipping=0.2,
            entropy_bonus=0.01):

        self.policy = BaseSoftmaxPolicy(
            env=function_approximator.env,
            update_strategy=update_strategy,
            ppo_clipping=ppo_clipping,
            entropy_bonus=entropy_bonus,
            random_seed=None,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.value_function = BaseV(
            env=function_approximator.env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=True,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.function_approximator = function_approximator
        self._init_models()

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='actor_critic/S', shape=shape, dtype=dtype)
        G = keras.Input(name='actor_critic/G', shape=[1], dtype='float')

        # shared part of the computation graph
        X = self.function_approximator.body(S, variable_scope='primary')
        X_target = self.function_approximator.body(S, variable_scope='target')

        # value head
        V = self.function_approximator.head_v(X, variable_scope='primary')
        V_target = self.function_approximator.head_v(
            X_target, variable_scope='target')

        # policy head (Z == logits)
        Z = self.function_approximator.head_pi(X, variable_scope='primary')
        Z_target = self.function_approximator.head_pi(
            X_target, variable_scope='target')

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
        value_loss = self.function_approximator.VALUE_LOSS_FUNCTION

        # joint train model
        self.train_model = keras.Model(inputs=[S, G], outputs=[Z, V])
        self.train_model.compile(
            loss=[policy_loss, value_loss],
            optimizer=self.function_approximator.optimizer)
        self.train_model.summary()
