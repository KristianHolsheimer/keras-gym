from abc import abstractmethod, ABC
import sys

import numpy as np
import scipy.stats as st

from gym.spaces.discrete import Discrete
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputClassifier

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
            assert P.shape == (1, self.env.action_space.n), "bad shape: P"
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


class BaseUpdateablePolicy(BasePolicy):
    """
    Base class for updateable policy objects, which are objects that can be
    updated directly, using e.g. policy-gradient type algorithms.

    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    classifier : sklearn classifier
        This classifier must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as :py:class:`FunctionTransformer
        <sklearn.preprocessing.FunctionTransformer>`. We can also use other
        transformers that only learn the input shape at training time, such as
        :py:class:`PolynomialFeatures
        <sklearn.preprocessing.PolynomialFeatures>`. Note that these do require
        us to set `attempt_fit_transformer=True`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    """
    def __init__(self, env, classifier, transformer=None,
                 attempt_fit_transformer=False, random_seed=None):

        BasePolicy.__init__(self, env, random_seed)

        self.classifier = classifier
        self.transformer = transformer
        self.attempt_fit_transformer = attempt_fit_transformer
        self._init_model()

    @abstractmethod
    def batch_eval(self, X_s):
        """
        Given a batch of preprocessed states, get the associated probabilities.

        .. note:: This has only been implemented for discrete action spaces.

        Parameters
        ----------
        X_s : array of float, shape = [batch_size, num_features]
            Preprocessed design matrix representing a batch of state
            observations. It is what comes out of :func:`X`.

        Returns
        -------
        params : 2d array, shape: [batch_size, num_params]
            The parameters required to describe the probability distribution
            over actions :math:`\\pi(a|s)`. For discrete action spaces,
            `params` is the array of probabilities
            :math:`(p_0, \\dots, p_{n-1})`, where :math:`p_i=P(a=i)`.

        """
        pass

    def update(self, X, Y):
        """
        Update the policy object function. This method will call
        :term:`partial_fit` on the underlying sklearn classifier.

        Parameters
        ----------
        X : 2d-array, shape = [batch_size, num_features]
            A sklearn-style design matrix of a single data point.

        Y : 1d- or 2d-array, depends on model type
            A sklearn-style label array. The shape depends on the model type.
            For a type-I model, the output shape is `[batch_size]` and for a
            type-II model the shape is `[batch_size, num_actions]`.

        """
        self.classifier.partial_fit(X, Y)

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
            classes = np.arange(self.env.action_space.n, dtype='int')
        except AttributeError:
            raise NonDiscreteActionSpaceError()

        # create dummy input X
        s = self.env.observation_space.sample()
        if isinstance(s, np.ndarray):
            s = np.random.rand(*s.shape)  # otherwise we get overflow

        X = self.X(s)
        Y = np.array([self.env.action_space.sample()])

        try:
            self.classifier.partial_fit(X, Y, classes=classes)
        except ValueError as e:
            expected_failure = (
                e.args[0].startswith("bad input shape") and  # Y has bad shape
                not isinstance(
                    self.classifier, MultiOutputClassifier))  # not yet wrapped
            if not expected_failure:
                raise
            self.classifier = MultiOutputClassifier(self.classifier)
            self.classifier.partial_fit(X, Y, classes=classes)
