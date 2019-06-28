import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class InteractionMixin:
    INTERACTION_OPTS = ('elementwise_quadratic', 'full_quadratic')

    @property
    def input_dim(self):
        if not hasattr(self, '_input_dim'):
            if not isinstance(self.env.observation_space, gym.spaces.Box):
                raise TypeError(
                    "expected observation space Box, got: {}"
                    .format(self.env.observation_space))
            self._input_dim = np.prod(self.env.observation_space.shape)
        return self._input_dim

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
        n = self.input_dim + 1  # input_dim needs to be known before first call
        indices = [[i, j] for i in range(n) for j in range(max(1, i), n)]
        return tf.gather_nd(tensor, indices)

    def _init_optimizer(self, optimizer, sgd_kwargs):
        if optimizer is None:
            self.optimizer = keras.optimizers.SGD(**sgd_kwargs)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "unknown optimizer, expected a keras.optimizers.Optimizer or "
                "None (which sets the default keras.optimizers.SGD optimizer)")
