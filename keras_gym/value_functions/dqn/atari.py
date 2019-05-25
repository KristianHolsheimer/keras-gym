import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ...utils import diff_transform_matrix
from ...losses import ProjectedSemiGradientLoss
from ...base.function_approximators.generic import GenericQTypeII


__all__ = (
    'AtariQ',
)


class AtariQ(GenericQTypeII):
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

        Keyword arguments for :class:`keras.optimizers.Adam`. See `keras docs
        <https://keras.io/optimizers/#adam>`_ for more details.

    """  # noqa: E501
    def __init__(
            self, env,
            gamma=0.99,
            bootstrap_n=1,
            update_strategy='q_learning',
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None,
            bootstrap_model=None)

        self._init_optimizer(optimizer, adam_kwargs)
        self._init_models()
        self._check_attrs()

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='S', shape=shape, dtype=dtype)
        G = keras.Input(name='G', shape=(), dtype='float')

        def diff_transform(S):
            S = K.cast(S, 'float32') / 255
            M = diff_transform_matrix(num_frames=shape[-1])
            return K.dot(S, M)

        def layers(variable_scope):
            def v(name):
                return '{}/{}'.format(variable_scope, name)

            return [
                keras.layers.Lambda(diff_transform),
                keras.layers.Conv2D(
                    name=v('conv1'), filters=16, kernel_size=8, strides=4,
                    activation='relu'),
                keras.layers.Conv2D(
                    name=v('conv2'), filters=32, kernel_size=4, strides=2,
                    activation='relu'),
                keras.layers.Flatten(name=v('flatten')),
                keras.layers.Dense(
                    name=v('dense1'), units=256, activation='relu'),
                keras.layers.Dense(
                    name=v('outputs'), units=self.num_actions,
                    kernel_initializer='zeros')]

        # forward pass
        def forward_pass(X, variable_scope):
            Y = X
            for layer in layers(variable_scope):
                Y = layer(Y)
            return Y

        # regular computation graph
        Q = forward_pass(S, variable_scope='primary')

        # loss
        loss = ProjectedSemiGradientLoss(G, base_loss=tf.losses.huber_loss)

        # regular models
        self.train_model = keras.Model(inputs=[S, G], outputs=Q)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Q)

        # target model
        Q_target = forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=Q_target)

        self.bootstrap_model = None

    def _init_optimizer(self, optimizer, adam_kwargs):
        if optimizer is None:
            self.optimizer = keras.optimizers.Adam(**adam_kwargs)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "unknown optimizer, expected a keras.optimizers.Optimizer or "
                "None (which sets the default keras.optimizers.Adam "
                "optimizer)")
