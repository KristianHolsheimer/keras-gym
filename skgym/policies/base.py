from __future__ import print_function, division
from abc import abstractmethod, ABC
import numpy as np
import scipy.stats as st
from gym.spaces.discrete import Discrete
from ..utils import argmax


class BasePolicy(ABC):
    """
    Abstract base class for policy objects.

    """
    def __init__(self, env, random_seed=None):
        self.env = env
        self.random_seed = random_seed

    @abstractmethod
    def X(self, s, a=None):
        pass

    @abstractmethod
    def X_next(self, s):
        pass

    @abstractmethod
    def batch_eval(self, *args):
        pass

    @abstractmethod
    def update(self, X, Y):
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
        X_s = self.X_next(s)
        P = self.batch_eval(X_s)
        if isinstance(self.env.action_space, Discrete):
            return st.multinomial(n=1, p=P[0])
        else:
            raise NotImplementedError(
                "I haven't yet implemented continuous action spaces; "
                "please send me a message to let me know if this is holding "
                "you back. -kris")

    def thompson(self, s):
        """
        Given a state observation :math:`s`, return an action :math:`a` drawn
        from the policy's probability distribution :math:`\\pi(a|s)` given by
        :func:`proba`.

        Parameters
        ----------
        s : state observation
            Depending on the observation space, `s` may be an integer or an
            array of floats.

        Returns
        -------
        a : action
            The action is sampled according to :math:`\\pi(a|s)`.

        """
        dist = self.proba(s)
        if isinstance(dist, st._multivariate.multinomial_frozen):
            a_onehot = dist.rvs(size=1)[0]
            a = np.argmax(a_onehot)  # int
        else:
            raise NotImplementedError(
                "I haven't yet implemented continuous action spaces; "
                "please send me a message to let me know if this is holding "
                "you back. -kris")
        return a

    def greedy(self, s):
        """
        Given a state observation :math:`s`, return an action :math:`a` in a
        way that is greedy with respect to the policy's probability
        distribution, given by :func:`proba`.

        Parameters
        ----------
        s : state observation
            Depending on the observation space, `s` may be an integer or an
            array of floats.

        Returns
        -------
        a : action
            The action with the highest probability under the current policy.

        """
        dist = self.proba(s)
        if isinstance(dist, st._multivariate.multinomial_frozen):
            p = dist.p
            a = argmax(p)
            return a
        elif isinstance(dist, st._distn_infrastructure.rv_frozen):  # Gaussian
            return dist.median()
        else:
            raise NotImplementedError(
                "I haven't yet implemented continuous action spaces; "
                "please send me a message to let me know if this is holding "
                "you back. -kris")

    def random(self):
        """
        Pick action uniformly at random.

        Returns
        -------
        a : action
            A random action.

        """
        return self.env.action_space.sample()

    def epsilon_greedy(self, s, epsilon=0.01):
        """
        Flip a epsilon-weighted coin to decide whether pick action
        using `greedy` or `random`.

        Parameters
        ----------
        s : int or array
            A single observation (state).

        epsilon : float
            The expected probability of picking a random action.

        Returns
        -------
        a : int
            An action, assuming discrete action space.

        """
        if self._random.rand() < epsilon:
            return self.random()
        else:
            return self.greedy(s)

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        self._random = np.random.RandomState(new_random_seed)
        self._random_seed = new_random_seed

    @random_seed.deleter
    def random_seed(self):
        self._random = np.random.RandomState(None)
        self._random_seed = None
