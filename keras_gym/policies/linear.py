from tensorflow import keras
from tensorflow.keras import backend as K

from ..losses import SoftmaxPolicyLossWithLogits
from ..base.function_approximators.linear import LinearFunctionMixin
from ..base.function_approximators.generic import GenericSoftmaxPolicy


__all__ = (
    'LinearSoftmaxPolicy',
)


class LinearSoftmaxPolicy(GenericSoftmaxPolicy, LinearFunctionMixin):
    def __init__(
            self, env,
            update_strategy='vanilla',
            interaction=None,
            optimizer=None,
            **sgd_kwargs):

        super().__init__(
            env=env,
            update_strategy=update_strategy,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self.interaction = interaction
        self._init_interaction_layer(interaction)
        self._init_optimizer(optimizer, sgd_kwargs)
        self._init_models(output_dim=self.num_actions)
        self._check_attrs()

    def _init_models(self, output_dim):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='S', shape=shape, dtype=dtype)
        Adv = keras.Input(name='Adv', shape=(), dtype='float')

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
        Logits = forward_pass(S, variable_scope='primary')

        # loss
        if self.update_strategy == 'vanilla':
            loss = SoftmaxPolicyLossWithLogits(Adv)
        elif self.update_strategy == 'trpo':
            raise NotImplementedError("update_strategy == 'trpo'")  # TODO
        elif self.update_strategy == 'ppo':
            raise NotImplementedError("update_strategy == 'ppo'")  # TODO
        else:
            raise ValueError(
                "unknown update_strategy '{}'".format(self.update_strategy))

        # regular models
        self.train_model = keras.Model(inputs=[S, Adv], outputs=Logits)
        self.train_model.compile(loss=loss, optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=Logits)

        # optional models
        self.target_model = None
