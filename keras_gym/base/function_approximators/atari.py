from abc import abstractmethod

from tensorflow import keras
from tensorflow.keras import backend as K

from ...utils import diff_transform_matrix, get_env_attr
from ...base.function_approximators.generic import BaseFunctionApproximator


__all__ = (
    'AtariFunctionMixin',
)


class AtariFunctionMixin(BaseFunctionApproximator):
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

        def diff_transform(S):
            num_frames = get_env_attr(self.env, 'num_frames')
            S = K.cast(S, 'float32') / 255
            M = diff_transform_matrix(num_frames=num_frames)
            return K.dot(S, M)

        layers = [
            keras.layers.Lambda(diff_transform, name=v('diff_transform')),
            keras.layers.Conv2D(
                name=v('conv1'), filters=16, kernel_size=8, strides=4,
                activation='relu'),
            keras.layers.Conv2D(
                name=v('conv2'), filters=32, kernel_size=4, strides=2,
                activation='relu'),
            keras.layers.Flatten(name=v('flatten')),
            keras.layers.Dense(
                name=v('dense1'), units=256, activation='relu')]

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
