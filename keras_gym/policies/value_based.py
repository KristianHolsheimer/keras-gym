from gym.spaces import Discrete

from .base import BasePolicy
from ..value_functions.base import BaseQ
from ..utils import softmax
from ..errors import NonDiscreteActionSpaceError


class ValuePolicy(BasePolicy):
    """
    Policies based on a state-action value function, i.e. it uses a Q-function
    to figure out what actions to pick.

    Parameters
    ----------
    value_function : Q-function object
        This Q-function must be a `BaseQ`-derived object.

    random_seed : int, optional
        Set a random state for reproducible randomization.

    """

    def __init__(self, value_function, random_seed=None):
        if not isinstance(value_function, BaseQ):
            raise TypeError("value_function must be a subtype of BaseQ")
        self.value_function = value_function
        super(ValuePolicy, self).__init__(value_function.env, random_seed)

    def __repr__(self):
        return (
            "ValuePolicy(value_function={value_function}, "
            "random_seed={_random_seed})".format(**self.__dict__))

    def batch_eval(self, X_s):
        """
        Given a batch of preprocessed states, get the associated probabilities.

        .. note:: This has only been implemented for discrete action spaces.

        Parameters
        ----------
        X_s : 2d or 3d array of float
            For a type-I value function the input shape is
            `[batch_size, num_features]` and for a type-II value function the
            input shape is `[num_actions, batch_size, num_features]`. It is
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
            (X_s.ndim == 3 and self.value_function.MODELTYPE == 1) or
            (X_s.ndim == 2 and self.value_function.MODELTYPE == 2))
        if not consistent:
            raise TypeError("shape and model type are inconsistent")

        if isinstance(self.env.action_space, Discrete):
            Q_s = self.value_function.batch_eval_typeII(X_s)
            params = softmax(Q_s, axis=1)
        else:
            raise NonDiscreteActionSpaceError()

        return params

    def X(self, s):
        """
        Create a feature vector from a state :math:`s` or state-action pair
        :math:`(s, a)`, depending on the model type.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X : 2d array
            Scikit-learn style design matrix.

        """
        return self.value_function.preprocess_typeII(s)
