from collections import deque
from itertools import islice

import numpy as np

from ..base.errors import InsufficientCacheError, EpisodeDoneError


__all__ = (
    'MonteCarloCache',
    'NStepCache',
)


class NStepCache:
    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)

        self._deque_sa = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = np.power(self.gamma, np.arange(self.n))
        self._gamman = np.power(self.gamma, self.n)

    def append(self, s, a, r, done):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly call popleft) before "
                "appending new transitions")

        self._deque_sa.append((s, a))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_sa)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def popleft(self):
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "popped from")

        # pop state-action pair
        s, a = self._deque_sa.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        self._deque_r.popleft()

        # keep in mind that we've already popped (s, a)
        if len(self) >= self.n:
            s_next, a_next = self._deque_sa[self.n - 1]
            i_next = self._gamman
        else:
            s_next, a_next, i_next = s, a, 0.  # no more bootstrapping

        S = np.array([s])
        A = np.array([a])
        Rn = np.array([rn])
        I_next = np.array([i_next])
        S_next = np.array([s_next])
        A_next = np.array([a_next])

        return S, A, Rn, I_next, S_next, A_next  # batch_size=1

    def flush(self):
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "flushed")

        S = []
        A = []
        Gn = []
        I_next = []
        S_next = []
        A_next = []

        while self:
            s, a, gn, i_next, s_next, a_next = self.popleft()
            S.append(s[0])
            A.append(a[0])
            Gn.append(gn[0])
            I_next.append(i_next[0])
            S_next.append(s_next[0])
            A_next.append(a_next[0])

        S = np.stack(S, axis=0)
        A = np.stack(A, axis=0)
        Gn = np.stack(Gn, axis=0)
        I_next = np.stack(I_next, axis=0)
        S_next = np.stack(S_next, axis=0)
        A_next = np.stack(A_next, axis=0)

        return S, A, Gn, I_next, S_next, A_next


class MonteCarloCache:
    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque = deque([])
        self._done = False
        self._g = 0  # accumulator for return

    def append(self, s, a, r, done):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly pop) before appending new "
                "transitions")

        self._deque.append((s, a, r))
        self._done = bool(done)
        if done:
            self._g = 0.  # init return

    def __len__(self):
        return len(self._deque)

    def __bool__(self):
        return bool(len(self)) and self._done

    def pop(self):
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be "
                    "popped from")
            else:
                raise InsufficientCacheError(
                    "cannot pop from cache before before receiving done=True")

        # pop state-action pair
        s, a, r = self._deque.pop()

        # update return
        self._g = r + self.gamma * self._g

        S = np.array([s])
        A = np.array([a])
        G = np.array([self._g])

        return S, A, G  # batch_size=1

    def flush(self):
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be "
                    "flushed")
            else:
                raise InsufficientCacheError(
                    "cannot flush cache before before receiving done=True")

        S = []
        A = []
        G = []

        while self:
            s, a, g = self.pop()
            S.append(s[0])
            A.append(a[0])
            G.append(g[0])

        S = np.stack(S, axis=0)
        A = np.stack(A, axis=0)
        G = np.stack(G, axis=0)

        return S, A, G
