import numpy as np
import pytest
from ..utils import ArrayDeque
from ..errors import ArrayDequeOverflowError


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
