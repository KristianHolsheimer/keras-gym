import numpy as np
from .errors import ArrayDequeOverflowError


def check_dtype(x, dtype):
    """
    Check the data type of a scalar value `x`.

    Parameters
    ----------
    x : scalar
        Input value.

    dtype : {'int', 'float'}
        The abstract datatype, i.e. numpy types of pure-python types.

    Returns
    -------
    is_type : bool
        Whether `x` is of type `dtype`.

    """
    if dtype == 'int':
        return (
            isinstance(x, (int, np.int_)) or
            (isinstance(x, (float, np.float_)) and float(x).is_integer()))

    if dtype == 'float':
        return isinstance(x, (float, np.float_))

    raise NotImplementedError("only implemented checks for 'int' and 'float'")


def idx(arr, axis=0):
    """
    Given a numpy array, return its corresponding integer index array.

    Parameters
    ----------
    arr : array
        Input array.

    axis : int, optional
        The axis along which we'd like to get an index.

    Returns
    -------
    index : 1d array, shape: arr.shape[axis]
        An index array `[0, 1, 2, ...]`.
    """
    return np.arange(arr.shape[axis])


def one_hot_vector(i, n, dtype='float'):
    """
    Create a dense one-hot encoded vector.

    Parameters
    ----------
    i : int
        The index of the non-zero entry.

    n : int
        The dimensionality of the dense vector. Note that `n` must be greater
        than `i`.

    dtype : str or datatype
        The output data type, default is `'float'`.

    Returns
    -------
    x : 1d array of length n
        The dense one-hot encoded vector.

    """
    if not 0 <= i < n:
        raise ValueError("i must be a non-negative and smaller than n")
    x = np.zeros(int(n), dtype=dtype)
    x[int(i)] = 1.0
    return x


def argmin(arr, axis=None, random_state=None):
    """
    This is a little hack to ensure that argmin breaks ties randomly, which is
    something that :func:`numpy.argmin` doesn't do.

    *Note: random tie breaking is only done for 1d arrays; for multidimensional
    inputs, we fall back to the numpy version.*

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.

    random_state : int or RandomState
        This can either be a random seed (`int`) or an instance of
        :class:`numpy.random.RandomState`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 1:
        candidates = np.arange(arr.size)             # all
        candidates = candidates[arr == np.min(arr)]  # min
        if not isinstance(random_state, np.random.RandomState):
            # treat input random_state as seed
            random_state = np.random.RandomState(random_state)
        return random_state.choice(candidates)
    else:
        return np.argmin(arr, axis=axis)


def argmax(arr, axis=None, random_state=None):
    """
    This is a little hack to ensure that argmax breaks ties randomly, which is
    something that :func:`numpy.argmax` doesn't do.

    *Note: random tie breaking is only done for 1d arrays; for multidimensional
    inputs, we fall back to the numpy version.*

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.

    random_state : int or RandomState
        This can either be a random seed (`int`) or an instance of
        :class:`numpy.random.RandomState`.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    return argmin(-arr, axis=axis, random_state=random_state)


def softmax(arr, axis=0):
    """
    Compute the softmax (normalized point-wise exponential).

    Parameters
    ----------
    arr : numpy array
        The input array.

    axis : int, optional
        The axis along which to normalize, default is 0.

    Returns
    -------
    out : array of same shape
        The entries of the output array are non-negative and normalized, which
        make them good candidates for modeling probabilities.

    """
    arr = np.clip(arr, -30, 30)
    arr = np.exp(arr)
    arr /= arr.sum(axis=axis, keepdims=True)
    return arr


class ArrayDeque:
    """
    A numpy array based deque, loosely based on the built-in class
    :py:class:`collections.deque`. Note that `ArrayDeque` has only partial
    functionality compared its built-in counterpart.

    Parameters
    ----------
    shape : array of ints
        The shape of the to-be-cached arrays. This is the shape of a single
        entry.

    maxlen : int, optional
        The maximum number of arrays that can be stored in the deque. Overflow
        strategy can be specified by the `overflow` argument.

    overflow : {'error', 'cycle', 'grow'}, optional
        What to do when `maxlen` is reached: `'error'` means raise an error,
        `'cycle'` means start overwriting old entries, and `'grow'` means grow
        `maxlen` by a factor of 2.

    dtype : str or datatype, optional
        The datatype of the to-be-cached arrays. Default is `'float'`.

    Attributes
    ----------
    array : numpy array
        Numpy array of stored values. The shape is
        `[num_arrays, <array_shape>]`.


    """
    def __init__(self, shape, maxlen=512, overflow='cycle', dtype='float'):
        if overflow not in ('error', 'cycle', 'grow'):
            raise ValueError("bad value for overflow")

        self.shape = tuple(shape)
        self.maxlen = maxlen
        self.dtype = dtype
        self.overflow = overflow

        self._len = 0
        self._idx = -1 % self.maxlen
        self._array = np.empty([maxlen] + list(shape), dtype)

    def __bool__(self):
        return bool(self._len)

    def __len__(self):
        return self._len

    def __str__(self):
        return (
            "ArrayDeque(shape={shape}, maxlen={maxlen}, dtype={dtype}, "
            "len={_len})".format(**self.__dict__))

    def __repr__(self):
        return str(self)

    @property
    def array(self):
        start = (self._idx - self._len + 1) % self.maxlen
        end = start + self._len
        if not self:
            return self._array[:0]
        indices = np.arange(start, end)
        return self._array.take(indices, axis=0, mode='wrap')

    def append(self, array):
        """
        Add an array to the deque.

        Parameters
        -----------
        array : numpy array
            The array to be added.

        """
        if self._len < self.maxlen:
            self._len += 1
        elif self.overflow == 'error':
            raise ArrayDequeOverflowError(
                "maxlen reached; consider increasing maxlen or set "
                "overflow='cycle' or overflow='grow'")
        elif self.overflow == 'cycle':
            pass
        elif self.overflow == 'grow':
            self._array = np.concatenate(
                [self._array, np.zeros_like(self._array)], axis=0)
            self.maxlen *= 2
            self._len += 1
            assert self.maxlen == self._array.shape[0], "{} != {}".format(
                self.maxlen, self._array.shape[0])

        self._idx = (self._idx + 1) % self.maxlen
        self._array[self._idx] = array

    def pop(self):
        """
        Pop the most recently added array from the deque.

        Returns
        -------
        array : numpy array
            The most recently added array.

        """
        if not self:
            raise IndexError("pop from an empty deque")
        value = self._array[self._idx]
        self._idx = (self._idx - 1) % self.maxlen
        self._len -= 1
        return value

    def popleft(self):
        """
        Pop the oldest array from the deque.

        Returns
        -------
        array : numpy array
            The oldest array.

        """
        if not self:
            raise IndexError("pop from an empty deque")
        self._len -= 1
        i = (self._idx - self._len) % self.maxlen
        value = self._array[i]
        return value
