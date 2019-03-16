from gym.spaces import Discrete

from .base import BasePolicy
from ..value_functions import GenericQ, GenericQTypeII
from ..utils import softmax
from ..errors import NonDiscreteActionSpaceError


class ValuePolicy(BasePolicy):
    """
    A simple policy derived from a state-action value function :math:`Q(s,a)`.

    In other words, this policy uses a Q-function to figure out what actions to
    pick.

    Parameters
    ----------
    value_function : Q-function object

        This Q-function must be a ``BaseQ``-derived object.

    random_seed : int, optional
        Set a random state for reproducible randomization.

    """

    def __init__(self, value_function, random_seed=None):
        if not isinstance(value_function, (GenericQ, GenericQTypeII)):
            raise TypeError(
                "value_function must be a subtype of BaseValueFunction")
        self.value_function = value_function
        super(ValuePolicy, self).__init__(value_function.env, random_seed)

    def __repr__(self):
        return (
            "ValuePolicy(value_function={value_function}, "
            "random_seed={_random_seed})".format(**self.__dict__))

    def batch_eval(self, X):
        """
        Given a batch of preprocessed states, get the associated probabilities.

        **Note:** This has only been implemented for discrete action spaces.

        Parameters
        ----------
        X : 2d or 3d array of float

            For a type-I value function the input shape is
            ``[batch_size, num_features]`` and for a type-II value function the
            input shape is ``[num_actions, batch_size, num_features]``. It is
            what comes out of :func:`X`.

        Returns
        -------
        params : 2d array, shape: [batch_size, num_params]
            The parameters required to describe the probability distribution
            over actions :math:`\\pi(a|s)`. For discrete action spaces,
            `params` is the array of probabilities
            :math:`(p_0, \\dots, p_{n-1})`, where :math:`p_i=P(a=i)`.

        """
        consistent = (
            (X.ndim == 3 and isinstance(self.value_function, GenericQ)) or
            (X.ndim == 2 and isinstance(self.value_function, GenericQTypeII))
        )
        if not consistent:
            raise TypeError("shape and model type are inconsistent")

        if isinstance(self.env.action_space, Discrete):
            Q = self.value_function.batch_eval_next(X)
            params = softmax(Q, axis=1)
        else:
            raise NonDiscreteActionSpaceError()

        return params

    def X(self, s):
        """
        Create a feature vector from a state :math:`s` or state-action pair
        :math:`(s, a)`, depending on the model type.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        X : 2d or 3d array of float

            For a value function that is derived from :class:`GenericQ
            <keras_gym.value_functions.GenericQ>` the output shape is
            ``[num_actions, batch_size, num_features]`` and if it is derived
            from :class:`GenericQTypeII
            <keras_gym.value_functions.GenericQTypeII>`, the output shape is
            ``[batch_size, num_features]``.

        """
        return self.value_function.X_next(s)
