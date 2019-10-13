from tensorflow import keras
from tensorflow.keras import backend as K

from ..generic import FunctionApproximator


__all__ = (
    'ConnectFourFunctionApproximator',
)


class ConnectFourFunctionApproximator(FunctionApproximator):
    """

    A :term:`function approximator` specifically designed for the
    :class:`ConnectFour <keras_gym.envs.ConnectFourEnv>` environment.

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
                name='conv1', filters=20, kernel_size=4, strides=1,
                activation='relu'),
            keras.layers.Conv2D(
                name='conv2', filters=40, kernel_size=2, strides=1,
                activation='relu'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(
                name='dense1', units=64, activation='linear'),
        ]

        # forward pass
        X = S
        for layer in layers:
            X = layer(X)

        return X
