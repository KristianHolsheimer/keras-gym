from tensorflow import keras
from tensorflow.keras import backend as K

from ..generic import FunctionApproximator


__all__ = (
    'ConnectFourFunctionApproximator',
)


class ConnectFourFunctionApproximator(FunctionApproximator):
    def body(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')

        def v(name):
            return '{}/{}'.format(variable_scope, name)

        def extract_state(S):
            return K.cast(S[:, 1:, :, :], 'float')

        def extract_available_actions_mask(S):
            return K.cast(S[:, 0, :, 0], 'bool')

        # extract the mask over available actions from the state observation
        self.available_actions_mask = keras.layers.Lambda(
            extract_available_actions_mask,
            name='extract_available_actions_mask')(S)

        layers = [
            keras.layers.Lambda(extract_state, name='extract_state'),
            keras.layers.Conv2D(
                name=v('conv1'), filters=20, kernel_size=4, strides=1,
                activation='relu'),
            keras.layers.Conv2D(
                name=v('conv2'), filters=40, kernel_size=2, strides=1,
                activation='relu'),
            keras.layers.Flatten(name=v('flatten')),
            keras.layers.Dense(
                name=v('dense1'), units=64, activation='linear'),
        ]

        # forward pass
        X = S
        for layer in layers:
            X = layer(X)

        return X
