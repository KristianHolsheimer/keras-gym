from tensorflow import keras

from ..base.function_approximators.generic import GenericV
from ..base.function_approximators.connect_four import ConnectFourFunctionMixin
from ..losses import LoglossSign


class ConnectFourV(GenericV, ConnectFourFunctionMixin):
    """
    A specific :term:`state value function` for the ConnectFour environment.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    gamma : float, optional

        The discount factor for discounting future rewards.

    bootstrap_n : positive int, optional

        The number of steps in n-step bootstrapping. It specifies the number of
        steps over which we're willing to delay bootstrapping. Large :math:`n`
        corresponds to Monte Carlo updates and :math:`n=1` corresponds to
        TD(0).

    bootstrap_with_target_model : bool, optional

        Whether to use the :term:`target_model` when constructing a
        bootstrapped target. If False (default), the primary
        :term:`predict_model` is used.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the Adam optimizer is used,
        :class:`keras.optimizers.Adam`. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **adam_kwargs : keyword arguments

        Keyword arguments for `keras.optimizers.Adam
        <https://keras.io/optimizers/#adam>`_.

    """
    def __init__(
            self, env,
            gamma=1.0,
            bootstrap_n=1,
            bootstrap_with_target_model=False,
            optimizer=None,
            **adam_kwargs):

        super().__init__(
            env=env,
            gamma=gamma,
            bootstrap_n=bootstrap_n,
            bootstrap_with_target_model=bootstrap_with_target_model,
            train_model=None,  # set models later
            predict_model=None,
            target_model=None)

        self._init_optimizer(optimizer, adam_kwargs)
        self._init_models()
        self._check_attrs()

    def _head(self, X, variable_scope):
        layer = keras.layers.Dense(
            units=1,
            kernel_initializer='zeros',
            name=(variable_scope + '/V'))
        return layer(X)

    def _init_models(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype

        S = keras.Input(name='value/S', shape=shape, dtype=dtype)

        # regular computation graph
        V = self._forward_pass(S, variable_scope='primary')

        # regular models
        self.train_model = keras.Model(inputs=S, outputs=V)
        self.train_model.compile(
            loss=LoglossSign(), optimizer=self.optimizer)
        self.predict_model = keras.Model(inputs=S, outputs=V)

        # target model
        V_target = self._forward_pass(S, variable_scope='target')
        self.target_model = keras.Model(inputs=S, outputs=V_target)
