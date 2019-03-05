from abc import ABC, abstractmethod
import sys

import numpy as np
from scipy.stats import multinomial
from sklearn.exceptions import NotFittedError
from gym.spaces import Discrete

from ..utils import softmax, feature_vector, RandomStateMixin
from ..errors import NonDiscreteActionSpaceError


class BasePolicy(ABC, RandomStateMixin):
    """


    Attributes
    ----------
    is_value_based : bool
        Whether the policy is a value-based policy (when value_function is
        specified) or not (when policy_regressor is specified).


    """
    def __init__(self, env, value_function=None, policy_regressor=None,
                 policy_transformer=None, random_seed=None):

        self.env = env
        self.random_seed = random_seed
        self.value_function = None
        self.policy_regressor = None
        self.policy_transformer = None
        self.is_value_based = None

        if value_function is not None:
            if policy_regressor is not None or policy_transformer is not None:
                raise ValueError(
                    "if value_function is provided, policy_regressor and "
                    "policy_transformer must be left unspecified")
            self.value_function = value_function
            self.is_value_based = True
        elif policy_regressor is not None:
            self.policy_regressor = policy_regressor
            self.policy_transformer = policy_transformer
            self.is_value_based = False
        else:
            raise ValueError(
                "must either specify value_function or policy_regressor "
                "(possibly with policy_transformer); cannot leave both "
                "unspecified")

    @abstractmethod
    def update(self, X, Y):
        """ Update the underlying sklearn-style function approximator. """
        pass

    @abstractmethod
    def __call__(self, s, return_propensity=False):
        """
        Draw the next action :math:`a` according to :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state
            The current state observation.

        Returns
        -------
        a or (a, p) : action or action-propensity pair

            The action `a` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If `return_propensity=True`, the
            propensity `p` is also returned, which is the probability of
            picking action `a` under the current policy.

        """
        pass

    def X(self, s):
        """
        Create a feature vector from a state-action pair.

        Parameters
        ----------
        s : int or array of float
            A single state observation.

        Returns
        -------
        X_s : 2d-array, shape = [1, num_features]
            A sklearn-style design matrix of a single data point.

        .. note::

            This method is used for policy-gradient type updates. For
            valuefunction updates, please use the value function's own methods
            instead.

        """
        if self.is_value_based:
            raise NotImplementedError(
                "This method is only implemented for policy-gradient type "
                "updates; for value function updates, please use the value "
                "function's own methods instead.")

        X_s = feature_vector(s, self.env.observation_space)
        X_s = np.expand_dims(X_s, axis=0)  # add batch axis (batch_size == 1)
        X_s = self._transform(X_s)  # apply transformer if provided
        return X_s

    def _transform(self, X):
        if self.transformer is not None:
            try:
                X = self.transformer.transform(X)
            except NotFittedError:
                if not self.attempt_fit_transformer:
                    raise NotFittedError(
                        "transformer needs to be fitted; setting "
                        "attempt_fit_transformer=True will fit the "
                        "transformer on one data point")
                print("attemting to fit transformer", file=sys.stderr)
                X = self.transformer.fit_transform(X)
        return X

    def _distr(self, X_s):
        """
        Given a batch of preprocessed state observation, return a batch of
        probability distributions over the space of actions
        :math:`\\mathcal{A}(s)` for each :math:`s` represented in the input
        batch.

        """
        if isinstance(self.env.action_space, Discrete):
            if self.is_value_based:
                Q_s = self.value_function.batch_eval_typeII(X_s)
                P_s = softmax(Q_s, axis=1)
            else:
                P_s = self.regressor.predict_proba(X_s)
            distr = np.array([multinomial(n=1, p=p) for p in P_s])
        else:
            raise NonDiscreteActionSpaceError()

        return distr


class RandomPolicy(BasePolicy):
    def __call__(self, s=None, return_propensity=False):
        if isinstance(self.env.action_space, Discrete):
            n = self.env.action_space.n
            a = self._random.randint(n)
            p = 1.0 / n
        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a

    def update(*args, **kwargs):
        pass


class ThompsonPolicy(BasePolicy):
    def __call__(self, s, return_propensity=False):
        if self.is_value_based:
            X_s = self.value_function.preprocess_typeII(s)
        else:
            X_s = self.X(s)
        distr = self._distr(X_s)[0]

        if isinstance(self.env.action_space, Discrete):
            a_onehot = distr.rvs(size=1).astype('bool').ravel()
            a = np.asscalar(np.argwhere(a_onehot))  # one-hot -> index
            p = distr.p[a]
        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a

    def update(*args, **kwargs):
        pass
