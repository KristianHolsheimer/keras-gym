from tensorflow import keras
from tensorflow.keras import backend as K

from ..generic import FunctionApproximator
from .mixins import InteractionMixin


__all__ = (
    'LinearFunctionApproximator',
)


class LinearFunctionApproximator(FunctionApproximator, InteractionMixin):
    """

    A linear :term:`function approximator`.

    Parameters
    ----------
    env : environment

        A gym-style environment.

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            'full_quadratic'
                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be the batch
                size and number of features. Then, the input shape is
                :math:`(b, n)` and the output shape is :math:`(b, (n + 1) (n
                + 2) / 2 - 1))`.

                **Note:** This option requires the `tensorflow` backend.

            'elementwise_quadratic'
                This option generates element-wise quadratic interactions,
                which only include linear and quadratic terms. It does *not*
                include bilinear terms or an intercept. Let :math:`b` and
                :math:`n` be the batch size and number of features. Then, the
                input shape is :math:`(b, n)` and the output shape is
                :math:`(b, 2n)`.

        Otherwise, a custom interaction layer can be passed as well. If left
        unspecified (`interaction=None`), the interaction layer is omitted
        altogether.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the function approximator's
        DEFAULT_OPTIMIZER is used. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **optimizer_kwargs : keyword arguments

        Keyword arguments for the optimizer. This is useful when you want to
        use the default optimizer with a different setting, e.g. changing the
        learning rate.

    """
    DEFAULT_OPTIMIZER = keras.optimizers.SGD

    def __init__(
            self, env,
            interaction=None,
            optimizer=None,
            **optimizer_kwargs):

        FunctionApproximator.__init__(self, env, optimizer, **optimizer_kwargs)
        self._init_interaction_layer(interaction)

    def body(self, S, variable_scope):
        if K.ndim(S) > 2:
            S = keras.layers.Flatten(name=(variable_scope + '/flatten'))(S)
        if self.interaction_layer is not None:
            S = self.interaction_layer(S)
        return S
