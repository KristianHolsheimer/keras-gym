import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .base import VFunction, QFunctionTypeI, QFunctionTypeII, Policy


__all__ = (
    'LinearV',
    'LinearQTypeI',
    'LinearQTypeII',
    'LinearPolicy',
)


class LinearFunctionMixin:
    INTERACTION_OPTS = ('elementwise_quadratic', 'full_quadratic')

    def _init_interaction_layer(self, interaction):
        if interaction is None:
            self.interaction_layer = None

        elif isinstance(interaction, keras.layers.Layer):
            self.interaction_layer = interaction

        elif interaction == 'elementwise_quadratic':
            self.interaction_layer = keras.layers.Lambda(
                self._elementwise_quadratic_interaction)

        elif interaction == 'full_quadratic':
            self.interaction_layer = keras.layers.Lambda(
                self._full_quadratic_interaction)
        else:
            raise ValueError(
                "unknown interaction, expected a keras.layers.Layer or a "
                "specific string, one of: {}".format(self.INTERACTION_OPTS))

    @staticmethod
    def _elementwise_quadratic_interaction(x):
        """

        This option generates element-wise quadratic interactions, which only
        include linear and quadratic terms. It does *not* include bilinear
        terms or an intercept. Let :math:`b` and :math:`n` be the batch size
        and number of features. Then, the input shape is :math:`(b, n)` and the
        output shape is :math:`(b, 2n)`.

        """
        x2 = K.concatenate([x, x ** 2])
        return x2

    def _full_quadratic_interaction(self, x):
        """

        This option generates full-quadratic interactions, which include all
        linear, bilinear and quadratic terms. It does *not* include an
        intercept. Let :math:`b` and :math:`n` be the batch size and number of
        features. Then, the input shape is :math:`(b, n)` and the output shape
        is :math:`(b, (n + 1) (n + 2) / 2 - 1))`.

        **Note:** This option requires the `tensorflow` backend.

        """
        ones = K.ones_like(K.expand_dims(x[:, 0], axis=1))
        x = K.concatenate([ones, x])
        x2 = tf.einsum('ij,ik->ijk', x, x)    # full outer product w/ dupes
        x2 = tf.map_fn(self._triu_slice, x2)  # deduped bi-linear interactions
        return x2

    def _triu_slice(self, tensor):
        """ Take upper-triangular slices to avoid duplicated features. """
        n = self.input_dim + 1  # needs to exists before first call
        indices = [[i, j] for i in range(n) for j in range(max(1, i), n)]
        return tf.gather_nd(tensor, indices)

    def _init_optimizer(self, optimizer, sgd_kwargs):
        if optimizer is None:
            self.optimizer = keras.optimizers.SGD(**sgd_kwargs)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "unknown optimizer, expected a keras.optmizers.Optimizer or "
                "None (which sets the default keras.optimizers.SGD optimizer)")


class LinearV(VFunction, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=0.9,
            interaction=None,
            optimizer=None,
            **sgd_kwargs):
        raise NotImplementedError('LinearV')


class LinearQTypeI(QFunctionTypeI, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            update_strategy='sarsa',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None,
            bootstrap_model=None)

        self.interaction = interaction
        self._init_interaction_layer(interaction)
        self._init_optimizer(optimizer, sgd_kwargs)
        self._init_models(output_dim=1)
        self._check_attrs()

    def _init_models(self, output_dim):
        s = self.env.observation_space.sample()

        S = keras.Input(name='S', shape=s.shape, dtype=s.dtype)
        A = keras.Input(name='A', shape=(), dtype='int32')

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
            loss=tf.losses.huber_loss, optimizer=self.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for type-I

        # optional models
        # Q_target = forward_pass(S, A, variable_scope='target')
        # self.target_model = keras.Model([S, A], Q_target)
        self.target_model = None
        self.bootstrap_model = None

    @staticmethod
    def _combine_state_action(S, A):
        return


class LinearQTypeII(QFunctionTypeII, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            update_strategy='q_learning',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):
        raise NotImplementedError('LinearQTypeII')


class LinearPolicy(Policy, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            update_strategy='vanilla',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):
        raise NotImplementedError('LinearPolicy')
