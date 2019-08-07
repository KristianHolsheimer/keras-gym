from abc import ABC, abstractmethod
from collections import deque
from itertools import islice

import numpy as np

from ..base.errors import InsufficientCacheError, EpisodeDoneError
from ..base.mixins import NumActionsMixin


__all__ = (
    'MonteCarloCache',
    'NStepCache',
)


class BaseShortTermCache(ABC, NumActionsMixin):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def add(self, s, pi, r, done):
        """
        Add a transition to the experience cache.

        Parameters
        ----------
        s : state observation

            A single state observation.

        pi : int or 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. This may
            be just an indicator if the action propensities are inferred
            through sampling. For instance, let's say our action space is
            :class:`Discrete(4)`, then passing ``pi = 2`` is equivalent to
            passing ``pi = [0, 0, 1, 0]``. Both would indicate that the action
            :math:`a=2` was drawn from the behavior policy.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        pass

    @abstractmethod
    def pop(self):
        """
        Pop a single transition from the cache.

        Returns
        -------
        #TODO

        """
        pass

    @abstractmethod
    def flush(self):
        """
        Flush all transitions from the cache.

        Returns
        -------
        #TODO

        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the cache to the initial state.

        """
        pass


class NStepCache(BaseShortTermCache):
    """
    A convenient helper class for n-step bootstrapping.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to determine ``num_actions``.

    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """
    def __init__(self, env, n, gamma):
        super().__init__(env)
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque_sa = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = np.power(self.gamma, np.arange(self.n))
        self._gamman = np.power(self.gamma, self.n)

    def add(self, s, pi, r, done):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly call popleft) before "
                "appending new transitions")
        pi = self.check_pi(pi)
        self._deque_sa.append((s, pi))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_sa)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        """
        Pop a single transition from the cache.

        Returns
        -------
        S, P, Rn, In, S_next, P_next : tuple of arrays, batch_size=1

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`P`, :term:`Rn`, :term:`In`, :term:`S_next`, :term:`P_next`)

            These are typically used for bootstrapped updates, e.g. minimizing
            the bootstrapped MSE:

            .. math::

                \\left(
                    R^{(n)}_t
                    + I^{(n)}_t\\,\\sum_aP(a|S_{t+n})\\,Q(S_{t+n},a)
                    - \\sum_aP(a|S_t)\\,Q(S_t,a) \\right)^2

        """  # noqa: E501
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "popped from")

        # pop state-action (propensities) pair
        s, pi = self._deque_sa.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        self._deque_r.popleft()

        # keep in mind that we've already popped (s, a)
        if len(self) >= self.n:
            s_next, pi_next = self._deque_sa[self.n - 1]
            i_next = self._gamman
        else:
            s_next, pi_next, i_next = s, pi, 0.  # no more bootstrapping

        S = np.array([s])
        P = np.array([pi])
        Rn = np.array([rn])
        In = np.array([i_next])
        S_next = np.array([s_next])
        P_next = np.array([pi_next])

        return S, P, Rn, In, S_next, P_next  # batch_size=1

    def flush(self):
        """
        Flush all transitions from the cache.

        Returns
        -------
        S, P, Rn, In, S_next, P_next : tuple of arrays

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`P`, :term:`Rn`, :term:`In`, :term:`S_next`, :term:`P_next`)

            These are typically used for bootstrapped updates, e.g. minimizing
            the bootstrapped MSE:

            .. math::

                \\left(
                    R^{(n)}_t
                    + I^{(n)}_t\\,\\sum_aP(a|S_{t+n})\\,Q(S_{t+n},a)
                    - \\sum_aP(a|S_t)\\,Q(S_t,a) \\right)^2

        """  # noqa: E501
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "flushed")

        S = []
        P = []
        Rn = []
        In = []
        S_next = []
        P_next = []

        while self:
            s, p, gn, i_next, s_next, p_next = self.pop()
            S.append(s[0])
            P.append(p[0])
            Rn.append(gn[0])
            In.append(i_next[0])
            S_next.append(s_next[0])
            P_next.append(p_next[0])

        S = np.stack(S, axis=0)
        P = np.stack(P, axis=0)
        Rn = np.stack(Rn, axis=0)
        In = np.stack(In, axis=0)
        S_next = np.stack(S_next, axis=0)
        P_next = np.stack(P_next, axis=0)

        return S, P, Rn, In, S_next, P_next


class MonteCarloCache(BaseShortTermCache):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._list = []
        self._done = False
        self._g = 0  # accumulator for return

    def add(self, s, pi, r, done):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly pop) before appending new "
                "transitions")

        pi = self.check_pi(pi)
        self._list.append((s, pi, r))
        self._done = bool(done)
        if done:
            self._g = 0.  # init return

    def __len__(self):
        return len(self._list)

    def __bool__(self):
        return bool(len(self)) and self._done

    def pop(self):
        """
        Pop a single transition from the cache.

        Returns
        -------
        S, P, G : tuple of arrays, batch_size=1

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`P`, :term:`G`)

        """
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be "
                    "popped from")
            else:
                raise InsufficientCacheError(
                    "cannot pop from cache before before receiving done=True")

        # pop state-action (propensities) pair
        s, pi, r = self._list.pop()

        # update return
        self._g = r + self.gamma * self._g

        S = np.array([s])
        P = np.array([pi])
        G = np.array([self._g])

        return S, P, G  # batch_size=1

    def flush(self):
        """
        Flush all transitions from the cache.

        Returns
        -------
        S, P, G : tuple of arrays

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`A`, :term:`G`)

        """
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be "
                    "flushed")
            else:
                raise InsufficientCacheError(
                    "cannot flush cache before before receiving done=True")

        S = []
        P = []
        G = []

        while self:
            s, pi, g = self.pop()
            S.append(s[0])
            P.append(pi[0])
            G.append(g[0])

        S = np.stack(S, axis=0)
        P = np.stack(P, axis=0)
        G = np.stack(G, axis=0)
        return S, P, G
