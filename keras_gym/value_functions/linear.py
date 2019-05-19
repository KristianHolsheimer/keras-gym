import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..base.function_approximators.linear import LinearFunctionMixin
from ..base.function_approximators.generic import (
    GenericV, GenericQTypeI, GenericQTypeII)
from ..losses import ProjectedSemiGradientLoss


__all__ = (
    'LinearV',
    'LinearQTypeI',
    'LinearQTypeII',
)


class LinearV(GenericV, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=0.9,
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
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
            loss=tf.losses.huber_loss, optimizer=self.optimizer)
        self.predict_model = self.train_model  # yes, it's trivial for V(s)

        # optional models
        # V_target = forward_pass(S, variable_scope='target')
        # self.target_model = keras.Model(S, V_target)
        self.target_model = None
        self.bootstrap_model = None


class LinearQTypeI(GenericQTypeI, LinearFunctionMixin):
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


class LinearQTypeII(GenericQTypeII, LinearFunctionMixin):
    def __init__(
            self, env,
            gamma=0.9,
            bootstrap_n=1,
            update_strategy='q_learning',
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
        self._init_models(output_dim=self.num_actions)
        self._check_attrs()

    def _init_models(self, output_dim):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='S', shape=shape, dtype=dtype)
        G = keras.Input(name='G', shape=(), dtype='float')

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
        loss = ProjectedSemiGradientLoss(G, base_loss=tf.losses.huber_loss)

        # regular models
        self.train_model = keras.Model(inputs=[S, G], outputs=Q)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Q)

        # optional models
        self.target_model = None
        self.bootstrap_model = None
