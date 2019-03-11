from tensorflow import keras

from ..value_functions.predefined import FeatureInteractionMixin
from .updateable import SoftmaxPolicy


class LinearPolicyMixin(FeatureInteractionMixin):

    def _optimizer(self, optimizer, **sgd_kwargs):
        if optimizer is None:
            return keras.optimizers.SGD(**sgd_kwargs)

        if isinstance(optimizer, keras.optimizers.Optimizer):
            return optimizer

        raise ValueError(
            "unknown optimizer, expected a keras.optmizers.Optimizer or "
            "None (which sets the default keras.optimizers.SGD optimizer)")

    def _model(self, output_size, interaction, optimizer, **sgd_kwargs):
        use_bias = (interaction != 'full_quadratic')
        model = keras.Sequential()
        if interaction is not None:
            model.add(self._interaction_layer(interaction))
        model.add(keras.layers.Dense(output_size, use_bias=use_bias))
        model.compile(
            loss=self._masked_mse_loss,
            optimizer=self._optimizer(optimizer, **sgd_kwargs),
        )
        return model


class LinearSoftmaxPolicy(SoftmaxPolicy):
    pass
