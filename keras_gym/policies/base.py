from abc import abstractmethod, ABC

import numpy as np
import scipy.stats as st

from gym.spaces.discrete import Discrete

from ..utils import argmax, RandomStateMixin, feature_vector
from ..errors import NonDiscreteActionSpaceError


class BasePolicy(ABC, RandomStateMixin):
    """
    Abstract base class for policy objects.

    Parameters
    ----------
    env : gym environment spec

        This is used to get information about the shape of the observation
        space and action space.

    random_seed : int, optional

        Set a random state for reproducible randomization.

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

            A single observation (state).

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

            A single observation (state).


        return_propensity : bool, optional

            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair

            The action ``a`` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If ``return_propensity=True``, the
            propensity ``p`` is also returned, which is the probability of
            picking action ``a`` under the current policy.

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

            A single observation (state).

        return_propensity : bool, optional

            Whether to return the propensity along with the drawn action. The
            propensity is the probability of picking the specific action to be
            returned.

        Returns
        -------
        a or (a, p) : action or action-propensity pair

            The action ``a`` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If ``return_propensity=True``, the
            propensity ``p`` is also returned, which is the probability of
            picking action ``a`` under the current policy.

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

            The action ``a`` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If ``return_propensity=True``, the
            propensity ``p`` is also returned, which is the probability of
            picking action ``a`` under the current policy.

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

            The action ``a`` is drawn from the probability distribution
            :math:`a\\sim\\pi(a|s)`. If ``return_propensity=True``, the
            propensity ``p`` is also returned, which is the probability of
            picking action ``a`` under the current policy.

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

    model : keras.Model

        A Keras function approximator. The input shape can be inferred from the
        data, but the output shape must be set to ``[1]``. Check out the
        :mod:`keras_gym.wrappers` module for wrappers that allow you to use
        e.g. scikit-learn function approximators instead.

    random_seed : int, optional

        Set a random state for reproducible randomization.

    """
    MODELTYPES = (2, 3)

    def __init__(self, env, model, random_seed=None):
        BasePolicy.__init__(self, env, random_seed)
        self.model = model
        self._check_model()

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
        params : 2d array, shape = [batch_size, num_params]

            The parameters required to describe the probability distribution
            over actions :math:`\\pi(a|s)`. For discrete action spaces,
            ``params`` is the array of probabilities
            :math:`(p_0, \\dots, p_{n-1})`, where :math:`p_i=P(a=i)`.

        """
        pass

    def update(self, X, A, advantages):
        """
        Update the policy object function. This method will call
        :term:`partial_fit` on the underlying sklearn classifier.

        Parameters
        ----------
        X : 2d-array, shape = [batch_size, num_features]

            A sklearn-style design matrix of a single data point.

        A : 1d-array, shape = [batch_size]

            A batch of actions taken.

        Y : 1d- or 2d-array, depends on model type

            A sklearn-style label array. The shape depends on the model type.
            For a type-I model, the output shape is ``[batch_size]`` and for a
            type-II model the shape is ``[batch_size, num_actions]``.

        """
        self.model.train_on_batch([X, advantages], A)

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

        """
        X_s = feature_vector(s, self.env.observation_space)
        X_s = np.expand_dims(X_s, axis=0)  # add batch axis (batch_size == 1)
        return X_s

    def _create_dummy_data(self):
        # n is needed to create dummy output Y
        try:
            n = self.env.action_space.n
        except AttributeError:
            raise NonDiscreteActionSpaceError()

        # sample a state observation from the environment
        s = self.env.observation_space.sample()
        if isinstance(s, np.ndarray):
            s = np.random.rand(*s.shape)  # otherwise we get overflow

        X = self.X(s)
        A = np.array([self.random()])
        advantages = np.zeros(1)

        # set some attributes for convenience
        # N.B. value_functions.predefined.Linear{V,Q} require these to be set
        self.num_features = X.shape[1]
        self.num_actions = n

        return X, A, advantages

    def _check_model(self):
        # get some dummy data
        X, A, advantages = self._create_dummy_data()

        weights_resettable = (
            hasattr(self.model, 'get_weights') and  # noqa: W504
            hasattr(self.model, 'set_weights'))

        if weights_resettable:
            weights = self.model.get_weights()

        try:
            self.update(X, A, advantages)
            pred = self.batch_eval(X)
            if self.MODELTYPE == 2:
                assert pred.shape == (1, self.num_actions), "bad shape"
                assert pred.sum() == 1.0, "niet normaaaaal"
            elif self.MODELTYPE == 3:
                # num_params = ...  # params distr over continuous actions
                # assert pred.shape == (num_params,), "bad model output shape"
                raise NotImplementedError("MODELTYPE == 3")
        except Exception as e:
            # TODO: show informative error message
            raise e

        if weights_resettable:
            self.model.set_weights(weights)
