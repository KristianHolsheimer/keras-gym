from abc import abstractmethod, ABC

import numpy as np
import scipy.stats as st

from gym.spaces.discrete import Discrete
from sklearn.exceptions import NotFittedError

from ..utils import argmax, RandomStateMixin, feature_vector
from ..errors import NonDiscreteActionSpaceError


class BasePolicy(ABC, RandomStateMixin):
    """
    Abstract base class for policy objects.

    """
    def __init__(self, env, random_seed=None):
        self.env = env
        self.random_seed = random_seed

    @abstractmethod
    def X(self, s):
        pass

    @abstractmethod
    def batch_eval(self, *args):
        pass

    def proba(self, s):
        """
        Given a state observation :math:`s`, return a proability distribution
        over all possible actions.

        Parameters
        ----------
        s : state observation
            Depending on the observation space, `s` may be an integer or an
            array of floats.

        Returns
        -------
        dist : scipy.stats probability distribution
            Depending on the action space, this may be a discrete distribution
            (typically a Dirichlet distribution) or a continuous distribution
            (typically a normal distribution).

        """
        X_s = self.X(s)
        P = self.batch_eval(X_s)
        if isinstance(self.env.action_space, Discrete):
            dist = st.multinomial(n=1, p=P[0], seed=self._random)
        else:
            raise NonDiscreteActionSpaceError()

        return dist

    def thompson(self, s, return_propensity=False):
        """
        Given a state observation :math:`s`, return an action :math:`a` drawn
        from the policy's probability distribution :math:`\\pi(a|s)` given by
        :func:`proba`.

        Parameters
        ----------
        s : state observation
            Depending on the observation space, `s` may be an integer or an
            array of floats.

        return_propensity : bool, optional
            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair
            The action `a` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If `return_propensity=True`, the
            propensity `p` is also returned, which is the probability of
            picking action `a` under the current policy.

        """
        dist = self.proba(s)
        if isinstance(self.env.action_space, Discrete):
            assert isinstance(dist, st._multivariate.multinomial_frozen)
            a_onehot = dist.rvs(size=1).ravel()
            a = np.asscalar(np.argmax(a_onehot))  # one-hot -> int
            p = dist.p[a]
        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a

    def greedy(self, s, return_propensity=False):
        """
        Given a state observation :math:`s`, return an action :math:`a` in a
        way that is greedy with respect to the policy's probability
        distribution, given by :func:`proba`.

        Parameters
        ----------
        s : state observation
            Depending on the observation space, `s` may be an integer or an
            array of floats.

        return_propensity : bool, optional
            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair
            The action `a` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If `return_propensity=True`, the
            propensity `p` is also returned, which is the probability of
            picking action `a` under the current policy.

        """
        dist = self.proba(s)
        if isinstance(self.env.action_space, Discrete):
            assert isinstance(dist, st._multivariate.multinomial_frozen)
            a = argmax(dist.p)
            p = 1.0
        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a

    def random(self, return_propensity=False):
        """
        Pick action uniformly at random.

        Parameters
        ----------
        return_propensity : bool, optional
            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair
            The action `a` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If `return_propensity=True`, the
            propensity `p` is also returned, which is the probability of
            picking action `a` under the current policy.

        """
        if isinstance(self.env.action_space, Discrete):
            n = self.env.action_space.n
            a = self._random.randint(n)
            p = 1.0 / n
        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a

    def epsilon_greedy(self, s, epsilon=0.01, return_propensity=False):
        """
        Flip a epsilon-weighted coin to decide whether pick action
        using `greedy` or `random`.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        epsilon : float
            The expected probability of picking a random action.

        return_propensity : bool, optional
            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair
            The action `a` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If `return_propensity=True`, the
            propensity `p` is also returned, which is the probability of
            picking action `a` under the current policy.

        """
        if isinstance(self.env.action_space, Discrete):
            a_greedy = self.greedy(s)
            n = self.env.action_space.n

            # draw action
            if self._random.rand() < epsilon:
                a = self.random()
            else:
                a = a_greedy

            # determine propensity
            if a == a_greedy:
                p = epsilon / n + (1 - epsilon)
            else:
                p = epsilon / n

        else:
            raise NonDiscreteActionSpaceError()

        return (a, p) if return_propensity else a


class BaseUpdateablePolicy(ABC, BasePolicy):
    def __init__(self, env, classifier, transformer=None,
                 attempt_fit_transformer=False, random_seed=None):

        super().__init__(env, random_seed)

        self.classifier = classifier
        self.transformer = transformer
        self.attempt_fit_transformer = attempt_fit_transformer
        self._init_model()

    @abstractmethod
    def update(self, X, Y):
        pass

    @abstractmethod
    def batch_eval(self, *args):
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

    def _init_model(self):
        # n is needed to create dummy output Y
        try:
            n = self.env.action_space.n
        except AttributeError:
            raise NonDiscreteActionSpaceError()

        # create dummy input X
        s = self.env.observation_space.sample()
        if isinstance(s, np.ndarray):
            s = np.random.rand(*s.shape)  # otherwise we get overflow

        X = self.X(s)
        Y = np.ones((1, n)) / n

        try:
            self.classifier.partial_fit(X, Y)
        except ValueError as e:
            expected_failure = (
                e.args[0].startswith("bad input shape") and  # Y has bad shape
                not isinstance(
                    self.classifier, MultiOutputClassifier))  # not yet wrapped
            if not expected_failure:
                raise
            self.classifier = MultiOutputClassifier(self.classifier)
            self.classifier.partial_fit(X, Y)
