from abc import abstractmethod

from tensorflow import keras
from tensorflow.keras import backend as K

from .generic import BaseFunctionApproximator


class ConnectFourFunctionMixin(BaseFunctionApproximator):
    @abstractmethod
    def _head(self, X, variable_scope):
        pass

    def _forward_pass(self, S, variable_scope):
        X = self._shared_forward_pass(S, variable_scope)
        Y = self._head(X, variable_scope)
        return Y

    def _shared_forward_pass(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')

        def v(name):
            return '{}/{}'.format(variable_scope, name)

        def extract_state(S):
            return K.cast(S[:, 1:, :, :], 'float')

        layers = [
            keras.layers.Lambda(extract_state, 'extract_state'),
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
        Y = S
        for layer in layers:
            Y = layer(Y)
        return Y

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
