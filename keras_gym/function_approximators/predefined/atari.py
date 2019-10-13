from tensorflow import keras
from tensorflow.keras import backend as K

from ...utils import diff_transform_matrix, get_env_attr
from ..generic import FunctionApproximator


__all__ = (
    'AtariFunctionApproximator',
)


class AtariFunctionApproximator(FunctionApproximator):
    """

    A :term:`function approximator` specifically designed for Atari 2600
    environments.

    Parameters
    ----------
    env : environment

        An Atari 2600 gym environment.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the function approximator's
        DEFAULT_OPTIMIZER is used. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **optimizer_kwargs : keyword arguments

        Keyword arguments for the optimizer. This is useful when you want to
        use the default optimizer with a different setting, e.g. changing the
        learning rate.

    """
    def body(self, S):

        def diff_transform(S):
            num_frames = get_env_attr(self.env, 'num_frames')
            S = K.cast(S, 'float32') / 255
            M = diff_transform_matrix(num_frames=num_frames)
            return K.dot(S, M)

        layers = [
            keras.layers.Lambda(diff_transform, name='diff_transform'),
            keras.layers.Conv2D(
                name='conv1', filters=16, kernel_size=8, strides=4,
                activation='relu'),
            keras.layers.Conv2D(
                name='conv2', filters=32, kernel_size=4, strides=2,
                activation='relu'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(
                name='dense1', units=256, activation='relu')]

        # forward pass
        X = S
        for layer in layers:
            X = layer(X)
        return X
