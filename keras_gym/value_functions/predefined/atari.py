import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from ...utils.atari import AtariPreprocessor
from ...losses import QTypeIIMeanSquaredErrorLoss
from ..generic import GenericQTypeII


class AtariDQN(GenericQTypeII):
    def __init__(
            self, env,
            target_model_sync_period=0,
            target_model_sync_tau=1.0):

        self.env = env
        self.preprocessor = AtariPreprocessor(num_frames=4)
        model = self._init_model()

        GenericQTypeII.__init__(
            self, env, model,
            target_model_sync_period=target_model_sync_period,
            target_model_sync_tau=target_model_sync_tau)

    def X(self, s):
        return np.reshape(self.preprocessor(s), (1, -1))

    def X_next(self, s_next):
        return np.reshape(self.preprocessor(s_next), (1, -1))

    def _init_model(self):

        # inputs
        X = keras.Input(name='X', shape=self.preprocessor.shape_flat)
        G = keras.Input(name='G', shape=[1])

        def reshape_and_rescale(x):
            shape = [-1] + list(self.preprocessor.shape)
            x = K.reshape(x / 255., shape)
            return x

        # sequential model
        layers = [
            keras.layers.Lambda(
                reshape_and_rescale, name='reshape_and_rescale'),
            keras.layers.Conv2D(
                name='conv1', filters=16, kernel_size=(8, 8), strides=(4, 4),
                activation='relu'),
            keras.layers.Conv2D(
                name='conv2', filters=32, kernel_size=(4, 4), strides=(2, 2),
                activation='relu'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(name='dense1', units=128, activation='relu'),
            keras.layers.Dense(
                name='outputs', units=self.num_actions,
                kernel_initializer='zeros'),
        ]

        # forward pass
        Y = X
        for layer in layers:
            Y = layer(Y)

        # model definition
        model = keras.Model(inputs=[X, G], outputs=Y)
        model.compile(
            loss=QTypeIIMeanSquaredErrorLoss(G),
            optimizer=keras.optimizers.RMSprop(
                lr=0.00025, rho=0.95, epsilon=0.01))

        return model
