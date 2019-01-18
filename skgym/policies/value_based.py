from __future__ import print_function, division
# import numpy as np

from .base import BasePolicy
from ..value_functions.base import BaseQ
from ..value_functions.generic import GenericQ
from ..utils import softmax


class ValueBasedPolicy(BasePolicy):
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
        super(ValueBasedPolicy, self).__init__(value_function.env, random_seed)

    def __repr__(self):
        return (
            "ValuePolicy(value_function={value_function}, "
            "random_seed={_random_seed})".format(**self.__dict__))

    def batch_eval(self, X_s):
        """
        Given a batch of preprocessed states, get the associated probabilities.

        Parameters
        ----------
        X_s : 2d or 3d array of float
            For a type-I value function the input shape is
            `[batch_size, num_features]` and for a type-II value function the
            input shape is `[num_actions, batch_size, num_features]`. It is
            what comes out of :func:`preprocess`.

        Returns
        -------
        Q_s : 2d array, shape: [batch_size, num_actions]

        """
        consistent = (
            (X_s.ndim == 3 and self.value_function.MODELTYPE == 1) or
            (X_s.ndim == 2 and self.value_function.MODELTYPE == 2))
        if not consistent:
            raise TypeError("shape and model type are inconsistent")

        Q_s = self.value_function.batch_eval_typeII(X_s)
        P = softmax(Q_s, axis=1)
        return P

    def X(self, s, a=None):
        """
        Create a feature vector from a state :math:`s` or state-action pair
        :math:`(s, a)`, depending on the model type.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        a : int, optional
            This is required for `model_type=1` and must be left unspecified
            for `model_type=2`.

        Returns
        -------
        X : 2d array
            Scikit-learn style design matrix.

        """

        return GenericQ.X(self.value_function, s, a)

    def X_next(self, s):
        """
        Create a feature vector from a state :math:`s` or state-action pair
        :math:`(s, a)`, depending on the model type.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        a : int, optional
            This is required for `model_type=1` and must be left unspecified
            for `model_type=2`.

        Returns
        -------
        X : 2d array
            Scikit-learn style design matrix.

        """
        return self.value_function.preprocess_typeII(s)

    def update(self, X, Y):
        self.value_function.update(X, Y)
