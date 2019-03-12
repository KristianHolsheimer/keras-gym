from tensorflow import keras

from ..value_functions.predefined import FeatureInteractionMixin
from ..losses import SoftmaxPolicyLossWithLogits
from .generic import GenericSoftmaxPolicy


class LinearPolicyMixin(FeatureInteractionMixin):

    def _optimizer(self, optimizer, **sgd_kwargs):
        if optimizer is None:
            return keras.optimizers.SGD(**sgd_kwargs)

        if isinstance(optimizer, keras.optimizers.Optimizer):
            return optimizer

        raise ValueError(
            "unknown optimizer, expected a keras.optmizers.Optimizer or "
            "None (which sets the default keras.optimizers.SGD optimizer)")

    def _model(self, output_dim, num_features, interaction, optimizer,
               **sgd_kwargs):

        # inputs
        X = keras.Input(shape=[num_features])
        advantages = keras.Input(shape=[1])

        # computation graph
        if interaction is not None:
            interaction_layer = self._interaction_layer(interaction)
            X = interaction_layer(X)
        dense = keras.layers.Dense(output_dim, kernel_initializer='zeros')
        logits = dense(X)

        # the final model
        model = keras.Model(inputs=[X, advantages], outputs=logits)
        model.compile(
            loss=SoftmaxPolicyLossWithLogits(advantages),
            optimizer=self._optimizer(optimizer, **sgd_kwargs))

        return model


class LinearSoftmaxPolicy(GenericSoftmaxPolicy, LinearPolicyMixin):
    """
    A linear function approximator for an updateable policy
    :math:`\\hat{\\pi}(a|s)`.

    This implementation uses a :class:`keras.Model` under the hood. Some simple
    feature interaction options are provided as well.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            interaction='full_quadratic'

                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be
                ``batch_size`` and ``num_features``, respectively. The input
                shape is :math:`(b,n)` and the output shape is :math:`(b,
                (n+1)(n+2)/2 - 1)`.

                .. note:: This option requires the Tensorflow backend.

            interaction='elementwise_quadratic'

                This option generates element-wise quadratic interactions,
                which only include linear and quadratic terms. It does *not*
                include bilinear terms or an intercept. Let :math:`b` and
                :math:`n` be ``batch_size`` and ``num_features``, respectively.
                The input shape is :math:`(b,n)` and the output shape is
                :math:`(b, 2n)`.

        Otherwise, a custom interaction layer can be passed as well. If left
        unspecified (``interaction=None``), the interaction layer is omitted
        altogether.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the plain vanilla SGD
        optimizer is used, :class:`keras.optimizers.SGD`. See `keras
        documentation <https://keras.io/optimizers/>`_ for more details.

    random_seed : int, optional

        Set a random state for reproducible randomization.

    sgd_kwargs : keyword arguments

        Keyword arguments for :class:`keras.optimizers.SGD`:

            lr : float >= 0
                Learning rate.

            momentum : float >= 0
                Parameter that accelerates SGD in the relevant direction and
                dampens oscillations.

            decay : float >= 0
                Learning rate decay over each update.

            nesterov : boolean
                Whether to apply Nesterov momentum.

        See `keras docs <https://keras.io/optimizers/#sgd>`_ for more details.

    """
    def __init__(self, env, interaction=None, optimizer=None, random_seed=None,
                 **sgd_kwargs):

        # need these dimensions to create function approximator
        self.env = env
        num_features = self.X(env.observation_space.sample()).shape[1]
        output_dim = env.action_space.n  # num_actions

        model = self._model(
            output_dim, num_features, interaction, optimizer, **sgd_kwargs)
        GenericSoftmaxPolicy.__init__(self, env, model, random_seed)
