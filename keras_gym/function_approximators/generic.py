import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..utils import check_tensor
from ..base.mixins import ActionSpaceMixin
from ..base.errors import ActionSpaceError


__all__ = (
    'FunctionApproximator',
)


class FunctionApproximator(ActionSpaceMixin):
    """
    A generic function approximator.

    This is the central object object that provides an interface between a
    gym-type environment and function approximators like :term:`value functions
    <state value function>` and :term:`updateable policies <updateable
    policy>`.

    In order to create a valid function approximator, you need to implement the
    :term:`body` method. For example, to implement a simple multi-layer
    perceptron function approximator you would do something like:

    .. code:: python

        import gym
        import keras_gym as km
        from tensorflow.keras.layers import Flatten, Dense

        class MLP(km.FunctionApproximator):
            \"\"\" multi-layer perceptron with one hidden layer \"\"\"
            def body(self, S):
                X = Flatten()(S)
                X = Dense(units=4)(X)
                return X

        # environment
        env = gym.make(...)

        # generic function approximator
        mlp = MLP(env, lr=0.001)

        # policy and value function
        pi, v = km.SoftmaxPolicy(mlp), km.V(mlp)

    The default :term:`heads <head>` are simple (multi) linear regression
    layers, which can be overridden by your own implementation.

    Parameters
    ----------
    env : environment

        A gym-style environment.

    optimizer : keras.optimizers.Optimizer, optional

        If left unspecified (``optimizer=None``), the function approximator's
        DEFAULT_OPTIMIZER is used. See `keras documentation
        <https://keras.io/optimizers/>`_ for more details.

    **optimizer_kwargs : keyword arguments

        Keyword arguments for the optimizer. This is useful when you want to
        use the default optimizer with a different setting, e.g. changing the
        learning rate.

    """
    DEFAULT_OPTIMIZER = keras.optimizers.Adam
    VALUE_LOSS_FUNCTION = keras.losses.Huber()

    def __init__(self, env, optimizer=None, **optimizer_kwargs):
        self.env = env
        self._init_optimizer(optimizer, optimizer_kwargs)

    def head_v(self, X):
        """
        This is the :term:`state value <state value function>` head. It returns
        a scalar V-value :math:`v(s)\\in\\mathbb{R}`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        V : 2d Tensor, shape: [batch_size, 1]

            The output :term:`state values <V>` :math:`v(s)\\in\\mathbb{R}`.

        """
        V = keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='zeros',
            name='value/v')(X)
        return V

    def head_q1(self, X):
        """
        This is the :term:`type-I <type-I state-action value function>`
        Q-value head. It returns a scalar Q-value
        :math:`q(s,a)\\in\\mathbb{R}`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Q_sa : 2d Tensor, shape: [batch_size, 1]

            The output :term:`type-I <Q_sa>` Q-values
            :math:`q(s,a)\\in\\mathbb{R}`.

        """
        Q_sa = keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='zeros',
            name='value/qtype1')(X)
        return Q_sa

    def head_q2(self, X):
        """
        This is the :term:`type-II <type-II state-action value function>`
        Q-value head. It returns a vector of Q-values
        :math:`q(s,.)\\in\\mathbb{R}^n`.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        Q_s : 2d Tensor, shape: [batch_size, num_actions]

            The output :term:`type-II <Q_s>` Q-values
            :math:`q(s,.)\\in\\mathbb{R}^n`.

        """
        Q_s = keras.layers.Dense(
            units=self.num_actions,
            activation='linear',
            kernel_initializer='zeros',
            name='value/qtype2')(X)
        return Q_s

    def head_pi(self, X):
        """
        This is the policy head. It returns logits, i.e. not probabilities. Use
        a softmax to turn the output into probabilities.

        Parameters
        ----------
        X : nd Tensor, shape: [batch_size, ...]

            ``X`` is an intermediate tensor in the full forward-pass of the
            computation graph; it's the output of the last layer of the
            :func:`body` method.

        Returns
        -------
        \\*params : Tensor or tuple of Tensors, shape: [batch_size, ...]

            These constitute the raw policy distribution parameters.

        """
        if self.action_space_is_discrete:
            logits = keras.layers.Dense(
                units=self.num_actions,
                activation='linear',
                kernel_initializer='zeros',
                name='policy/logits')(X)
            return logits

        if self.action_space_is_box:
            mu = keras.layers.Dense(
                units=self.actions_ndim,
                activation='linear',
                kernel_initializer='zeros',
                name='policy/mu')(X)
            logvar = keras.layers.Dense(
                units=self.actions_ndim,
                activation='linear',
                kernel_initializer='zeros',
                name='policy/logvar')(X)
            return mu, logvar

        raise ActionSpaceError.feature_request(self.env)

    def body(self, S):
        """
        This is the part of the computation graph that may be shared between
        e.g. policy (actor) and value function (critic). It is typically the
        part of a neural net that does most of the heavy lifting. One may think
        of the :func:`body` as an elaborate automatic feature extractor.

        Parameters
        ----------
        S : nd Tensor: shape: [batch_size, ...]

            The input state observation.

        Returns
        -------
        X : nd Tensor, shape: [batch_size, ...]

            The intermediate keras tensor.

        """
        def to_vector(x):
            if K.ndim(x) == 1 and K.dtype(x).startswith('int'):
                x = K.one_hot(x, self.env.observation_space.n)
            elif K.ndim(S) > 2:
                x = keras.layers.Flatten()(x)
            return x

        return keras.layers.Lambda(to_vector)(S)

    def body_q1(self, S, A):
        """
        This is similar to :func:`body`, except that it takes a state-action
        pair as input instead of only state observations.

        Parameters
        ----------
        S : nd Tensor: shape: [batch_size, ...]

            The input state observation.

        A : nd Tensor: shape: [batch_size, ...]

            The input actions.

        Returns
        -------
        X : nd Tensor, shape: [batch_size, ...]

            The intermediate keras tensor.

        """
        def kronecker_product(args):
            S, A = args
            if K.ndim(S) == 1 and K.dtype(S).startswith('int'):
                S = K.one_hot(S, self.env.observation_space.n)
            elif K.ndim(S) > 2:
                S = keras.layers.Flatten()(S)
            check_tensor(S, ndim=2, dtype=('float32', 'float64'))
            check_tensor(A, ndim=2, dtype=('float32', 'float64'))
            return tf.einsum('ij,ik->ijk', S, A)

        X = keras.layers.Lambda(kronecker_product)([S, A])
        X = keras.layers.Flatten()(X)
        return X

    def _init_optimizer(self, optimizer, optimizer_kwargs):
        if optimizer is None:
            self.optimizer = self.DEFAULT_OPTIMIZER(**optimizer_kwargs)
        elif isinstance(optimizer, keras.optimizers.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "unknown optimizer, expected a keras.optimizers.Optimizer or "
                "None (which sets the default keras.optimizers.Adam "
                "optimizer)")
