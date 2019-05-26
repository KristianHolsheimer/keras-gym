from itertools import islice

import pytest
import numpy as np

from ..base.errors import InsufficientCacheError, EpisodeDoneError
from .short_term import NStepCache, MonteCarloCache


class TestNStepCache:
    gamma = 0.85
    n = 5

    # rnd = np.random.RandomState(42)
    # S = np.arange(13)
    # A = rnd.randint(10, size=13)
    # R = rnd.randn(13)
    # D = np.zeros(13, dtype='bool')
    # D[-1] = True
    # I_next = (gamma ** n) * np.ones(13, dtype='bool')
    # I_next[-n:] = 0

    S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    A = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7])
    R = np.array([-0.48, 0.16, 0.23, 0.11, 1.46, 1.53, -2.43, 0.60, -0.25,
                  -0.16, -1.47, 1.48, -0.02])
    D = np.array([False] * 12 + [True])
    I_next = np.array([0.44370531249999995] * 8 + [0.0] * 5)
    episode = list(zip(S, A, R, D))

    @property
    def Rn(self):
        Rn_ = np.zeros_like(self.R)
        gammas = np.power(self.gamma, np.arange(13))
        for i in range(len(Rn_)):
            Rn_[i] = self.R[i:(i + self.n)].dot(
                gammas[:len(self.R[i:(i + self.n)])])
        return Rn_

    def test_append_done_twice(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            if i == 1:
                cache.add(s, a, r, True)
            else:
                with pytest.raises(EpisodeDoneError):
                    cache.add(s, a, r, True)

    def test_append_done_one(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            if i == 1:
                cache.add(s, a, r, True)
            else:
                break

        assert cache
        S, A, Rn, I_next, S_next, A_next = cache.flush()
        np.testing.assert_array_equal(S, self.S[:1])
        np.testing.assert_array_equal(A, self.A[:1])
        np.testing.assert_array_equal(Rn, self.R[:1])
        np.testing.assert_array_equal(I_next, [0])

    def test_pop(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i
            if i <= self.n:
                assert not cache
            if i > self.n:
                assert cache

        i = 0
        while cache:
            s, a, rn, i_next, s_next, a_next = cache.pop()
            assert s == self.S[i]
            assert a == self.A[i]
            assert rn == self.Rn[i]
            assert i_next == self.I_next[i]
            if i < 13 - self.n:
                assert s_next == self.S[i + self.n]
                assert a_next == self.A[i + self.n]
            i += 1

    def test_pop_eager(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode):
            cache.add(s, a, r, done)
            assert len(cache) == min(i + 1, self.n + 1)

            if cache:
                assert i + 1 > self.n
                s, a, gn, i_next, s_next, a_next = cache.pop()
                assert s == self.S[i - self.n]
                assert a == self.A[i - self.n]
                assert gn == self.Rn[i - self.n]
                assert i_next == self.I_next[i - self.n]
                assert s_next == self.S[i]
                assert a_next == self.A[i]
            else:
                assert i + 1 <= self.n

        i = 13 - self.n
        while cache:
            s, a, gn, i_next, s_next, a_next = cache.pop()
            assert s == self.S[i]
            assert a == self.A[i]
            assert gn == self.Rn[i]
            assert i_next == self.I_next[i]
            if i < 13 - self.n:
                assert s_next == self.S[i + self.n]
                assert a_next == self.A[i + self.n]
            i += 1

    def test_flush(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i
            if i <= self.n:
                assert not cache
            if i > self.n:
                assert cache

        S, A, Rn, I_next, S_next, A_next = cache.flush()
        np.testing.assert_array_equal(S, self.S)
        np.testing.assert_array_equal(A, self.A)
        np.testing.assert_array_equal(Rn, self.Rn)
        np.testing.assert_array_equal(I_next, self.I_next)
        np.testing.assert_array_equal(S_next[:-self.n], self.S[self.n:])
        np.testing.assert_array_equal(A_next[:-self.n], self.A[self.n:])

    def test_flush_eager(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode):
            cache.add(s, a, r, done)
            assert len(cache) == min(i + 1, self.n + 1)

            if cache:
                assert i + 1 > self.n
                S, A, Rn, I_next, S_next, A_next = cache.flush()
                if i == 12:
                    slc = slice(i - self.n, None)
                    np.testing.assert_array_equal(S, self.S[slc])
                    np.testing.assert_array_equal(A, self.A[slc])
                    np.testing.assert_array_equal(Rn, self.Rn[slc])
                    np.testing.assert_array_equal(I_next, self.I_next[slc])
                    np.testing.assert_array_equal(S_next.shape, (self.n + 1,))
                    np.testing.assert_array_equal(A_next.shape, (self.n + 1,))
                else:
                    slc = slice(i - self.n, i - self.n + 1)
                    np.testing.assert_array_equal(S, self.S[slc])
                    np.testing.assert_array_equal(A, self.A[slc])
                    np.testing.assert_array_equal(Rn, self.Rn[slc])
                    np.testing.assert_array_equal(I_next, self.I_next[slc])
                    np.testing.assert_array_equal(S_next, self.S[i])
                    np.testing.assert_array_equal(A_next, self.A[i])
            else:
                assert i + 1 <= self.n

        i = 13 - self.n
        while cache:
            s, a, gn, i_next, s_next, a_next = cache.pop()
            assert s == self.S[i]
            assert a == self.A[i]
            assert gn == self.Rn[i]
            assert i_next == self.I_next[i]
            if i < 13 - self.n:
                assert s_next == self.S[i + self.n]
                assert a_next == self.A[i + self.n]
            i += 1

    def test_flush_insufficient(self):
        cache = NStepCache(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in islice(enumerate(self.episode, 1), 4):
            cache.add(s, a, r, done)

        with pytest.raises(InsufficientCacheError):
            cache.flush()

    def test_flush_empty(self):
        cache = NStepCache(self.n, gamma=self.gamma)

        with pytest.raises(InsufficientCacheError):
            cache.flush()


class TestMonteCarloCache:
    gamma = 0.85
    S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    A = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7])
    R = np.array([-0.48, 0.16, 0.23, 0.11, 1.46, 1.53, -2.43, 0.60, -0.25,
                  -0.16, -1.47, 1.48, -0.02])
    D = np.array([False] * 12 + [True])
    G = np.zeros_like(R)
    for i, r in enumerate(R[::-1]):
        G[i] = r + gamma * G[i - 1]
    G = G[::-1]
    episode = list(zip(S, A, R, D))

    def test_append_pop_too_soon(self):
        cache = MonteCarloCache(self.gamma)
        for s, a, r, done in self.episode:
            cache.add(s, a, r, done)
            break

        with pytest.raises(InsufficientCacheError):
            cache.pop()

    def test_append_pop_expected(self):
        cache = MonteCarloCache(self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i

        assert cache
        assert len(cache) == 13

        for i in range(13):
            assert cache
            s, a, g = cache.pop()
            assert self.S[12 - i] == s
            assert self.A[12 - i] == a
            assert self.G[12 - i] == g

        assert not cache

    def test_append_flush_too_soon(self):
        cache = MonteCarloCache(self.gamma)
        for i, (s, a, r, done) in islice(enumerate(self.episode, 1), 4):
            cache.add(s, a, r, done)
            assert len(cache) == i

        with pytest.raises(InsufficientCacheError):
            cache.flush()

    def test_append_flush_expected(self):
        cache = MonteCarloCache(self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i

        S, A, G = cache.flush()
        np.testing.assert_array_equal(S, self.S[::-1])
        np.testing.assert_array_equal(A, self.A[::-1])
        np.testing.assert_array_equal(G, self.G[::-1])
