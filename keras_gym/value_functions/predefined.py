import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..losses import (
    SemiGradientMeanSquaredErrorLoss, QTypeIIMeanSquaredErrorLoss)
from .generic import GenericV, GenericQ, GenericQTypeII


class FeatureInteractionMixin:
    INTERACTION_OPTS = ('elementwise_quadratic', 'full_quadratic')

    def _interaction_layer(self, interaction):
        if isinstance(interaction, keras.layers.Layer):
            return interaction

        if interaction == 'elementwise_quadratic':
            return keras.layers.Lambda(self._elementwise_quadratic_interaction)

        if interaction == 'full_quadratic':
            return keras.layers.Lambda(self._full_quadratic_interaction)

        raise ValueError(
            "unknown interaction, expected a keras.layers.Layer or a specific "
            "string, one of: {}".format(self.INTERACTION_OPTS))

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
        n = self.input_dim + 1  # needs to exists before first call
        indices = [[i, j] for i in range(n) for j in range(max(1, i), n)]
        return tf.gather_nd(tensor, indices)


class LinearValueFunctionMixin(FeatureInteractionMixin):

    def _optimizer(self, optimizer, **sgd_kwargs):
        if optimizer is None:
            return keras.optimizers.SGD(**sgd_kwargs)

        if isinstance(optimizer, keras.optimizers.Optimizer):
            return optimizer

        raise ValueError(
            "unknown optimizer, expected a keras.optmizers.Optimizer or "
            "None (which sets the default keras.optimizers.SGD optimizer)")

    def _models(self, interaction, optimizer, **sgd_kwargs):

        # inputs
        X = keras.Input(name='X', shape=[self.input_dim])
        X_next = keras.Input(name='X_next', shape=[self.input_dim])
        I_next = keras.Input(name='I_next', shape=[1])

        def forward_pass(X):
            if interaction is not None:
                interaction_layer = self._interaction_layer(interaction)
                X = interaction_layer(X)
            dense_layer = keras.layers.Dense(
                self.output_dim, kernel_initializer='zeros')
            y = dense_layer(X)
            return y

        # output values
        y = forward_pass(X)
        bootstrapped_values = I_next * forward_pass(X_next)

        # the main model
        model = keras.Model(inputs=X, outputs=y)
        model.compile(
            loss=keras.losses.mse,
            optimizer=self._optimizer(optimizer, **sgd_kwargs),
        )

        # enable bootstrap updates
        bootstrap_model = keras.Model(inputs=[X, X_next, I_next], outputs=y)
        bootstrap_model.compile(
            loss=SemiGradientMeanSquaredErrorLoss(bootstrapped_values),
            optimizer=self._optimizer(optimizer, **sgd_kwargs),
        )

        return model, bootstrap_model


class LinearV(GenericV, LinearValueFunctionMixin):
    """
    A linear function approximator for :math:`V(s)`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            interaction = 'full_quadratic'

                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be the batch
                size and number of features. Then, the input shape is
                :math:`(b, n)` and the output shape is :math:`(b, (n + 1) (n
                + 2) / 2 - 1))`.

                **Note:** This option requires the `tensorflow` backend.

            interaction = 'elementwise_quadratic'

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

        If left unspecified (``optimizer=None``), the plain vanilla SGD optimizer
        is used, :class:`keras.optimizers.SGD`. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

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
    def __init__(self, env, interaction=None, optimizer=None, **sgd_kwargs):
        self._set_env_and_input_dim(env)
        model, bootstrap_model = self._models(
            interaction, optimizer, **sgd_kwargs)
        GenericV.__init__(self, env, model, bootstrap_model)


class LinearQ(GenericQ, LinearValueFunctionMixin):
    """
    A linear function approximator for :math:`Q(s, a)`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    state_action_combiner : {'outer', 'concatenate'} or func

        How to combine the feature vectors coming from ``s`` and ``a``. Here
        'outer' means taking a flat outer product using :py:func:`numpy.kron`,
        which gives a 1d-array of length :math:`d_s\\times d_a`. This choice is
        suitable for simple linear models, including the table-lookup type
        models. In contrast, 'concatenate' uses :py:func:`numpy.hstack` to
        return a 1d array of length :math:`d_s + d_a`. This option is more
        suitable for non-linear models.

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            interaction = 'full_quadratic'

                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be the batch
                size and number of features. Then, the input shape is
                :math:`(b, n)` and the output shape is :math:`(b, (n + 1) (n
                + 2) / 2 - 1))`.

                **Note:** This option requires the `tensorflow` backend.

            interaction = 'elementwise_quadratic'

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

        If left unspecified (``optimizer=None``), the plain vanilla SGD optimizer
        is used, :class:`keras.optimizers.SGD`. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

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
    def __init__(self, env, state_action_combiner='outer', interaction=None,
                 optimizer=None, **sgd_kwargs):

        self._init_combiner(state_action_combiner)
        self._set_env_and_input_dim(env)

        model, bootstrap_model = self._models(
            interaction, optimizer, **sgd_kwargs)

        GenericQ.__init__(
            self, env, model, bootstrap_model, state_action_combiner)


class LinearQTypeII(GenericQTypeII, LinearValueFunctionMixin):
    """
    A linear function approximator for :math:`Q(s, .)`.

    This type of model is different from the regular :class:`LinearQ` in that
    it models the Q-function differently. That is, instead of mapping
    :math:`(s, a)\\mapsto Q(s, a)` it maps :math:`s\\mapsto Q(s, .)`. Thus,
    it only takes a state observation :math:`s` as input and maps it to vector
    whose length is given by the number of actions.

    **Note**: This object assumes that the action space is discrete, i.e.
    :class:`gym.spaces.Discrete`.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    interaction : str or keras.layers.Layer, optional

        The desired feature interactions that are fed to the linear regression
        model. Available predefined preprocessors can be chosen by passing a
        string, one of the following:

            interaction = 'full_quadratic'

                This option generates full-quadratic interactions, which
                include all linear, bilinear and quadratic terms. It does *not*
                include an intercept. Let :math:`b` and :math:`n` be the batch
                size and number of features. Then, the input shape is
                :math:`(b, n)` and the output shape is :math:`(b, (n + 1) (n
                + 2) / 2 - 1))`.

                **Note:** This option requires the `tensorflow` backend.

            interaction = 'elementwise_quadratic'

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

        If left unspecified (``optimizer=None``), the plain vanilla SGD optimizer
        is used, :class:`keras.optimizers.SGD`. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

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
    def __init__(self, env, interaction=None, optimizer=None, **sgd_kwargs):
        self._set_env_and_input_dim(env)
        model, bootstrap_model = self._models(
            interaction, optimizer, **sgd_kwargs)
        GenericQTypeII.__init__(self, env, model, bootstrap_model)

    def _models(self, interaction, optimizer, **sgd_kwargs):

        # inputs
        X = keras.Input(name='X', shape=[self.input_dim])
        G = keras.Input(name='G', shape=[1])

        def forward_pass(X):
            if interaction is not None:
                interaction_layer = self._interaction_layer(interaction)
                X = interaction_layer(X)
            dense_layer = keras.layers.Dense(
                self.output_dim, kernel_initializer='zeros')
            y = dense_layer(X)
            return y

        # output values
        y = forward_pass(X)

        # the main model
        model = keras.Model(inputs=[X, G], outputs=y)
        model.compile(
            loss=QTypeIIMeanSquaredErrorLoss(G),
            optimizer=self._optimizer(optimizer, **sgd_kwargs),
        )

        return model, None
