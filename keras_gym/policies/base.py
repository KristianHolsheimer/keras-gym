from abc import abstractmethod, ABC

import gym
import numpy as np
import scipy.stats as st

from gym.spaces.discrete import Discrete

from ..utils import argmax, RandomStateMixin, feature_vector
from ..errors import NonDiscreteActionSpaceError, BadModelOuputShapeError


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


class EnvironmentDimensionsMixin:
    @property
    def num_actions(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        else:
            raise NonDiscreteActionSpaceError()

    def _set_env_and_input_dim(self, env):
        self.env = env

        # create dummy X
        s = self.env.observation_space.sample()
        X = self.X(s)

        # avoid overflow in model (space.sample can return very large numbers)
        X = (X - X.min()) / (X.max() - X.min())

        # set attribute
        self.input_dim = X.shape[1]

        return X


class BaseUpdateablePolicy(BasePolicy, EnvironmentDimensionsMixin):
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
    def __init__(self, env, model, random_seed=None):
        self._set_env_and_input_dim(env)
        BasePolicy.__init__(self, env, random_seed)
        self.model = model
        self._check_model()

    @abstractmethod
    def batch_eval(self, X_s):
        """
        Given a batch of preprocessed states, get the associated probabilities.

        **Note**: This has only been implemented for discrete action spaces.

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

        advantages : 1d-array, shape = [batch_size]

            This is the input that is either computed with a value function
            (critic) or with Monte Carlo type averaging.

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

    def _set_env_and_input_dim(self, env):
        self.env = env

        # create dummy X
        s = self.env.observation_space.sample()
        X = self.X(s)

        # avoid overflow in model (space.sample can return very large numbers)
        X = (X - X.min()) / (X.max() - X.min())

        # set attribute
        self.input_dim = X.shape[1]

        return X

    def _check_model(self):
        # get some dummy data
        X = self._set_env_and_input_dim(self.env)
        A = np.array([self.env.action_space.sample()])
        advantages = np.zeros(1)

        weights_resettable = (
            hasattr(self.model, 'get_weights') and  # noqa: W504
            hasattr(self.model, 'set_weights'))

        if weights_resettable:
            weights = self.model.get_weights()

        self.update(X, A, advantages)
        pred = self.batch_eval(X)
        if pred.shape != (1, self.output_dim):
            raise BadModelOuputShapeError((1, self.output_dim), pred.shape)

        if weights_resettable:
            self.model.set_weights(weights)


class BaseActorCritic(EnvironmentDimensionsMixin):
    def __init__(self, policy, value_function, train_model=None):
        self.policy = policy
        self.value_function = value_function
        self.train_model = train_model
        self._check_train_model()

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

    def update(self, X, A, Gn, X_next, I_next):
        """
        Update both policy (actor) and value function (critic).

        Parameters
        ----------

        X : 2d-array, shape = [batch_size, num_features]

            Preprocessed state observation :math:`X=\phi(s)`.

        A : 1d-array, shape = [batch_size]

            Batch of chosen actions.

        Gn : 1d-array, shape = [batch_size]

            Batch of (partial) returns.

        X_next : 2d-array, shape = [batch_size, num_features]

            Preprocessed state observation
            :math:`X_\\texttt{next}=\phi(s_\\texttt{next})`. This can be used
            to compute a bootstrapped target.

        I_next : 1d-array, shape [batch_size]

            A batch of discount factors. For instance, in n-step bootstrapping
            this is given by :math:`I_\\textt{next}=\\gamma^n`.


        """
        if self.target_func_update_delay == 0 and self.train_model is not None:
            self.train_model.train_on_batch([X, Gn, X_next, I_next], A)
        else:
            V = self.value_function.batch_eval(X)
            if self.target_func is not None:
                V_next = self.target_func.batch_eval_next(X_next)
            else:
                V_next = self.value_function.batch_eval_next(X_next)
            G = Gn + I_next * V_next
            advantages = G - V
            self.value_function.update(X, G)
            self.policy.update(X, A, advantages)

    def _check_train_model(self):
        if self.train_model is None:
            return

        # get some dummy data
        X = self._set_env_and_input_dim(self.env)
        A = np.array([self.env.action_space.sample()])
        Gn = np.random.randn(1)
        X_next = self._set_env_and_input_dim(self.env)
        I_next = np.random.rand(1)

        weights_resettable = (
            hasattr(self.train_model, 'get_weights') and  # noqa: W504
            hasattr(self.train_model, 'set_weights'))

        if weights_resettable:
            weights = self.train_model.get_weights()

        self.update(X, A, Gn, X_next, I_next)
        policy_pred = self.policy.batch_eval(X)
        value_pred = self.value_function.batch_eval(X)
        if policy_pred.shape != (1, self.output_dim):
            raise BadModelOuputShapeError(
                (1, self.output_dim), policy_pred.shape)
        if value_pred.shape != (1,):
            raise BadModelOuputShapeError((1,), value_pred.shape)

        if weights_resettable:
            self.train_model.set_weights(weights)
