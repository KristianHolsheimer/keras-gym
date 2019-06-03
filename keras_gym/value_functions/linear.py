import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..base.function_approximators.linear import LinearFunctionMixin
from ..base.function_approximators.generic import (
    GenericV, GenericQTypeI, GenericQTypeII)
from ..losses import ProjectedSemiGradientLoss, Huber


__all__ = (
    'LinearV',
    'LinearQTypeI',
    'LinearQTypeII',
)


class LinearV(GenericV, LinearFunctionMixin):
    """
    Linear-model implementation of a :term:`state value function`.

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

        If left unspecified (``optimizer=None``), the plain vanilla `SGD
        <https://keras.io/optimizers/#sgd>`_ optimizer is used. See `keras
        documentation <https://keras.io/optimizers/>`_ for other options.

    **sgd_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.SGD
        <https://keras.io/optimizers/#sgd>`_.

    """
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=0.9,
            bootstrap_with_target_model=False,
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.interaction = interaction
        self._init_interaction_layer(interaction)
        self._init_optimizer(optimizer, sgd_kwargs)
        self._init_models(output_dim=1)
        self._check_attrs()

    def _init_models(self, output_dim):
        s = self.env.observation_space.sample()

        S = keras.Input(name='value/S', shape=s.shape, dtype=s.dtype)

        def forward_pass(S, variable_scope):
            def v(name):
                return '{}/{}'.format(variable_scope, name)

            if K.ndim(S) > 2:
                S = keras.layers.Flatten(S)

            S = keras.layers.Flatten()(S)

            if self.interaction_layer is not None:
                S = self.interaction_layer(S)

            dense_layer = keras.layers.Dense(
                output_dim, kernel_initializer='zeros', name=v('weights'))

            return dense_layer(S)

        # regular models
        Q = forward_pass(S, variable_scope='primary')
        self.train_model = keras.Model(S, Q)
        self.train_model.compile(
            loss=Huber(), optimizer=self.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for V(s)

        # target model
        V_target = forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(S, V_target)


class LinearQTypeI(GenericQTypeI, LinearFunctionMixin):
    """
    Linear-model implementation of a :term:`type-I <type-I state-action value
    function>` Q-function.

    A :term:`type-I <type-I state-action value function>` Q-function is
    implemented by mapping :math:`(s, a)\\mapsto Q(s,a)`.

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

        If left unspecified (``optimizer=None``), the plain vanilla `SGD
        <https://keras.io/optimizers/#sgd>`_ optimizer is used. See `keras
        documentation <https://keras.io/optimizers/>`_ for other options.

    **sgd_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.SGD
        <https://keras.io/optimizers/#sgd>`_.

    """
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='sarsa',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.interaction = interaction
        self._init_interaction_layer(interaction)
        self._init_optimizer(optimizer, sgd_kwargs)
        self._init_models(output_dim=1)
        self._check_attrs()

    def _init_models(self, output_dim):
        s = self.env.observation_space.sample()

        S = keras.Input(name='value/S', shape=s.shape, dtype=s.dtype)
        A = keras.Input(name='value/A', shape=(), dtype='int32')

        def forward_pass(S, A, variable_scope):
            def v(name):
                return '{}/{}'.format(variable_scope, name)

            def kron(args):
                S, A = args
                A = tf.one_hot(A, self.num_actions)
                return tf.einsum('ij,ik->ijk', S, A)

            if K.ndim(S) > 2:
                S = keras.layers.Flatten(S)

            X = keras.layers.Lambda(kron)([S, A])
            X = keras.layers.Flatten()(X)

            if self.interaction_layer is not None:
                X = self.interaction_layer(X)

            dense_layer = keras.layers.Dense(
                output_dim, kernel_initializer='zeros', name=v('weights'))

            return dense_layer(X)

        # regular models
        Q = forward_pass(S, A, variable_scope='primary')
        self.train_model = keras.Model(inputs=[S, A], outputs=Q)
        self.train_model.compile(
            loss=Huber(), optimizer=self.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for type-I

        # target model
        Q_target = forward_pass(S, A, variable_scope='target')
        self.target_model = keras.Model([S, A], Q_target)


class LinearQTypeII(GenericQTypeII, LinearFunctionMixin):
    """
    Linear-model implementation of a :term:`type-II <type-II state-action value
    function>` Q-function.

    A :term:`type-II <type-II state-action value function>` Q-function is
    implemented by mapping :math:`s\\mapsto Q(s,.)`.

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

        If left unspecified (``optimizer=None``), the plain vanilla `SGD
        <https://keras.io/optimizers/#sgd>`_ optimizer is used. See `keras
        documentation <https://keras.io/optimizers/>`_ for other options.

    **sgd_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.SGD
        <https://keras.io/optimizers/#sgd>`_.

    """
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            update_strategy='q_learning',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
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

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)
        G = keras.Input(name='value/G', shape=(), dtype='float')

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
        Q = forward_pass(S, variable_scope='primary')

        # loss
        loss = ProjectedSemiGradientLoss(G, base_loss=Huber())

        # regular models
        self.train_model = keras.Model(inputs=[S, G], outputs=Q)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Q)

        # target model
        Q_target = forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=Q_target)
