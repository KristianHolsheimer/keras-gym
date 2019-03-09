import numpy as np
import pytest
from ..utils import ArrayDeque, ExperienceCache, softmax
from ..errors import ArrayDequeOverflowError


def test_softmax():
    rnd = np.random.RandomState(7)
    w = rnd.randn(3, 5)
    x = softmax(w, axis=1)
    y = softmax(w + 100., axis=1)
    z = softmax(w * 100., axis=1)

    # check shape
    assert x.shape == w.shape

    # check normalization
    np.testing.assert_almost_equal(x.sum(axis=1), np.ones(3))

    # check translation invariance
    np.testing.assert_almost_equal(y.sum(axis=1), np.ones(3))
    np.testing.assert_almost_equal(x, y)

    # check robustness by clipping
    assert not np.any(np.isnan(z))
    np.testing.assert_almost_equal(z.sum(axis=1), np.ones(3))


class TestArrayDeque:

    def test_cycle(self):
        d = ArrayDeque(shape=[], maxlen=3, overflow='cycle')
        d.append(1)
        d.append(2)
        d.append(3)
        np.testing.assert_array_equal(d.array, [1, 2, 3])

        d.append(4)
        np.testing.assert_array_equal(d.array, [2, 3, 4])

        assert d.pop() == 4
        np.testing.assert_array_equal(d.array, [2, 3])

        d.append(5)
        np.testing.assert_array_equal(d.array, [2, 3, 5])

        d.append(6)
        np.testing.assert_array_equal(d.array, [3, 5, 6])

        assert d.popleft() == 3
        np.testing.assert_array_equal(d.array, [5, 6])

        assert d.pop() == 6
        assert len(d) == 1
        assert d

        assert d.pop() == 5
        assert len(d) == 0
        assert not d

        with pytest.raises(IndexError, match="pop from an empty deque"):
            d.pop()

        with pytest.raises(IndexError, match="pop from an empty deque"):
            d.popleft()

    def test_error(self):
        d = ArrayDeque(shape=[], maxlen=3, overflow='error')
        d.append(1), d.append(2), d.append(3)
        np.testing.assert_array_equal(d.array, [1, 2, 3])

        with pytest.raises(ArrayDequeOverflowError):
            d.append(4)

    def test_grow(self):
        d = ArrayDeque(shape=[], maxlen=3, overflow='grow')
        d.append(1), d.append(2), d.append(3)
        np.testing.assert_array_equal(d.array, [1, 2, 3])

        d.append(4)
        np.testing.assert_array_equal(d.array, [1, 2, 3, 4])

        assert d.pop() == 4
        np.testing.assert_array_equal(d.array, [1, 2, 3])

        assert d.popleft() == 1
        assert d.pop() == 3
        assert d.popleft() == 2
        assert not d

        with pytest.raises(IndexError, match="pop from an empty deque"):
            d.pop()

        with pytest.raises(IndexError, match="pop from an empty deque"):
            d.popleft()


class TestExperienceCache:

    @staticmethod
    def create_obj(seed, length):
        rnd = np.random.RandomState(seed)
        ec = ExperienceCache()
        for i in range(length):
            X = rnd.randn(1, 5)
            X[0, 0] = i
            A = rnd.randint(4, size=1)
            R = rnd.randn(1)
            X_next = rnd.randn(1, 5)
            ec.append(X, A, R, X_next)
        return ec

    def test_pop(self):
        ec = self.create_obj(seed=13, length=7)
        assert len(ec) == 7

        X, A, R, X_next = ec.pop()
        assert len(ec) == 6
        np.testing.assert_array_almost_equal(
            X, [[6.0, -0.072247,  0.122339, -0.126747, -0.735211]])
        np.testing.assert_array_almost_equal(R, [0.318789])
        np.testing.assert_array_almost_equal(A, [0])
        np.testing.assert_array_almost_equal(
            X_next, [[-1.38233, 1.486765, 1.580075, 1.75342, 0.529771]])

    def test_popleft(self):
        ec = self.create_obj(seed=13, length=7)
        assert len(ec) == 7

        X, A, R, X_next = ec.popleft()
        assert len(ec) == 6
        np.testing.assert_array_almost_equal(
            X, [[0., 0.753766, -0.044503, 0.451812, 1.345102]])
        np.testing.assert_array_almost_equal(R, [0.532338])
        np.testing.assert_array_almost_equal(A, [2])
        np.testing.assert_array_almost_equal(
            X_next, [[-1.58899, -1.114284, 0.683527, 0.272894, -0.230848]])

    def test_popleft_nstep(self):
        n = 4
        ec = self.create_obj(seed=13, length=7)
        assert len(ec) == 7

        X, A, R, X_next = ec.popleft_nstep(n)
        assert R.shape == (4,)
        assert len(ec) == 6
        np.testing.assert_array_almost_equal(
            X, [[0., 0.753766, -0.044503, 0.451812, 1.345102]])
        np.testing.assert_array_almost_equal(
            R, [0.532338, 0.466986, 1.476737, 0.490872])
        np.testing.assert_array_almost_equal(A, [2])
        np.testing.assert_array_almost_equal(
            X_next, [[-1.38233, 1.486765, 1.580075, 1.75342, 0.529771]])

        X, A, R, X_next = ec.popleft_nstep(n)
        assert R.shape == (4,)
        assert len(ec) == 5
        assert X[0, 0] == 1

        X, A, R, X_next = ec.popleft_nstep(n)
        assert R.shape == (4,)
        assert len(ec) == 4
        assert X[0, 0] == 2
        assert X_next is not None

        X, A, R, X_next = ec.popleft_nstep(n)
        assert R.shape == (4,)
        assert len(ec) == 3
        assert X[0, 0] == 3
        assert X_next is not None

        X, A, R, X_next = ec.popleft_nstep(n)
        assert R.shape == (3,)
        assert len(ec) == 2
        assert X_next is None
        np.testing.assert_array_almost_equal(
            X, [[4., -1.512845, -0.764034, 0.10127, -0.317266]])
        np.testing.assert_array_almost_equal(
            R, [1.138333, -1.069344, 0.318789])
        np.testing.assert_array_almost_equal(A, [3])
