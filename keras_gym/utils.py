import numpy as np
from gym.spaces import Tuple, Discrete, Box, MultiDiscrete, MultiBinary

from .errors import ArrayDequeOverflowError, NoExperienceCacheError


def reload_all():
    from importlib import import_module, reload
    modules = (
        'keras_gym.errors',
        'keras_gym.utils',
        'keras_gym.losses',
        'keras_gym.wrappers.sklearn',
        'keras_gym.wrappers',
        'keras_gym.value_functions.base',
        'keras_gym.value_functions.generic',
        'keras_gym.value_functions.predefined',
        'keras_gym.value_functions',
        'keras_gym.policies.base',
        'keras_gym.policies.value_based',
        'keras_gym.policies.generic',
        'keras_gym.policies.predefined',
        'keras_gym.policies',
        'keras_gym.environments.adversarial',
        'keras_gym.environments',
        'keras_gym.algorithms.base',
        'keras_gym.algorithms.td0',
        'keras_gym.algorithms.monte_carlo',
        'keras_gym.algorithms.nstep_bootstrap',
        'keras_gym.algorithms',
    )
    for m in modules:
        reload(import_module(m))


def feature_vector(x, space):
    if isinstance(space, Tuple):
        x = np.concatenate([
            feature_vector(x_, space_)  # recursive
            for x_, space_ in zip(x, space.spaces)], axis=0)
    elif isinstance(space, MultiDiscrete):
        x = np.concatenate([
            feature_vector(x_, Discrete(n))  # recursive
            for x_, n in zip(x.ravel(), space.nvec.ravel()) if n], axis=0)
    elif isinstance(space, Discrete):
        x = one_hot_vector(x, space.n)
    elif isinstance(space, (MultiBinary, Box)):
        pass
    else:
        raise NotImplementedError(
            "haven't implemented a preprocessor for space type: {}"
            .format(type(space)))

    assert x.ndim == 1, "x must be 1d array, got shape: {}".format(x.shape)
    return x


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
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr -= arr.mean(axis=axis, keepdims=True)  # center before clipping
    arr = np.clip(arr, -30, 30)                # avoid overflow before exp
    arr = np.exp(arr)
    arr /= arr.sum(axis=axis, keepdims=True)
    return arr


def accumulate_rewards(R, gamma=1.0):
    """
    Accumulate individual rewards collected from an entire episode into
    gamma-discounted returns.

    Given an input array or rewards :math:`\\textbf{R} = [R_0, R_1, R_2,
    \\dots, R_{n-1}]`, then the generated output will be :math:`\\textbf{G} =
    [G_0, G_1, G_2, \\dots, G_{n-1}]`, where

    .. math::

        G_0 &= R_0+\\gamma R_1+\\gamma^2 R_2+\\dots+\\gamma^{n-1}R_{n-1}\\\\
        G_1 &= R_1+\\gamma R_2+\\gamma^2 R_3+\\dots+\\gamma^{n-2}R_{n-2}\\\\
        G_2 &= R_2+\\gamma R_3+\\gamma^2 R_4+\\dots+\\gamma^{n-3}R_{n-3}\\\\
        &\\vdots \\\\
        G_{n-3} &= R_{n-3}+\\gamma R_{n-2}+\\gamma^2 R_{n-1}\\\\
        G_{n-2} &= R_{n-2}+\\gamma R_{n-1}\\\\
        G_{n-1} &= R_{n-1}


    Parameters
    ----------
    R : 1d-array of float
        The individual rewards collected over an entire episode.

    Returns
    -------
    G : 1d-array of float
        The gamma-discounted cumulative rewards, i.e. returns, for the given
        episode.

    """
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    assert R.ndim == 1, "bad input shape"
    if gamma == 1.0:
        return np.cumsum(R[::-1])[::-1]  # much faster

    # use a custom ufunc.accumulate
    uadd = np.frompyfunc(lambda G, r: r + gamma * G, 2, 1)
    R_rev = R[::-1].astype('O')
    G_rev = uadd.accumulate(R_rev)
    G = G_rev[::-1].astype('float')
    return G


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
        self.overflow = overflow
        self.dtype = dtype

        self._len = 0
        self._maxlen = maxlen
        self._idx = -1 % self.maxlen
        self._array = np.empty([maxlen] + list(shape), dtype)

    def clear(self, reset_maxlen=True):
        """
        Clear the deque and optionally reset maxlen to its original setting.

        Parameters
        ----------
        reset_maxlen : bool, optional
            Whether to reset maxlen to its original setting. This is only
            applicable when `overflow` is set to `'grow'`.

        """
        maxlen_keep = self.maxlen
        self.__init__(self.shape, self._maxlen, self.overflow, self.dtype)
        if not reset_maxlen:
            self.maxlen = maxlen_keep

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


class RandomStateMixin:
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


class ExperienceCache(RandomStateMixin):
    """
    A class for conveniently storing and replaying experience. Each unit of
    experience consists of a preprocessed transition `(X, A, R, X_next)`.

    Parameters
    ----------
    maxlen : int, optional
        The maximum number of arrays that can be stored in the deque. Overflow
        strategy can be specified by the `overflow` argument.

    overflow : {'error', 'cycle', 'grow'}, optional
        What to do when `maxlen` is reached: `'error'` means raise an error,
        `'cycle'` means start overwriting old entries, and `'grow'` means grow
        `maxlen` by a factor of 2.

    Attributes
    ----------
    X_ : ArrayDeque
        Dict of numpy arrays of stored values. The shape of each value is
        `[num_arrays, <array_shape>]`.

    """
    def __init__(self, maxlen=512, overflow='cycle', random_seed=None):
        self.maxlen = maxlen
        self.overflow = overflow
        self.random_seed = random_seed
        self._len = 0

    def __len__(self):
        return len(self.X_)

    def __bool__(self):
        return bool(self.X_)

    def clear(self):
        if not self._check_fitted(raise_error=False):
            return
        self.X_.clear()
        self.A_.clear()
        self.R_.clear()
        self.X_next_.clear()

    def append(self, X, A, R, X_next):
        """
        Add a preprocessed transition to the experience cache.

        .. note::

            This method has only been implemented for `batch_size=1`.

        Parameters
        ----------
        X : 2d-array, shape: [batch_size, num_features]
            Scikit-learn style design matrix. This represents a batch of either
            states or state-action pairs, depending on the model type.

        A : 1d-array, shape: [batch_size]
            A batch of actions taken.

        R : 1d-array, shape: [batch_size]
            A batch of observed rewards.

        X_next : 2d-array, shape depends on model type
            The preprocessed next-state feature vector.

        """

        # check imput shapes (batch_size == 1)
        assert X.ndim > 0 and X.shape[0] == 1
        assert A.ndim > 0 and A.shape[0] == 1
        assert R.ndim > 0 and R.shape[0] == 1
        assert X_next.ndim > 0 and X_next.shape[0] == 1

        # create cache objects
        if not self._check_fitted(raise_error=False):
            self.X_ = ArrayDeque(
                shape=X.shape[1:], dtype=X.dtype,
                overflow=self.overflow, maxlen=self.maxlen)
            self.A_ = ArrayDeque(
                shape=A.shape[1:], dtype=A.dtype,
                overflow=self.overflow, maxlen=self.maxlen)
            self.R_ = ArrayDeque(
                shape=R.shape[1:], dtype=R.dtype,
                overflow=self.overflow, maxlen=self.maxlen)
            self.X_next_ = ArrayDeque(
                shape=X_next.shape[1:], dtype=X_next.dtype,
                overflow=self.overflow, maxlen=self.maxlen)

        # add to cache
        self.X_.append(X[0])
        self.A_.append(A[0])
        self.R_.append(R[0])
        self.X_next_.append(X_next[0])

    def sample(self, n=1):
        """
        Draw a sample, uniformly at random.

        Parameters
        ----------
        n : int
            The sample size.

        Returns
        -------
        X, A, R, X_next : arrays
            A batch of preprocessed transitions. The size of the first axis of
            each of the four arrays matches and are equal to `batch_size=n`.

        """
        if not isinstance(n, (int, np.int_)) or n <= 0:
            raise TypeError("n must be a positive integer")
        idx = self._random.permutation(np.arange(len(self)))[:n]
        X = self.X_.array[idx]
        A = self.A_.array[idx]
        R = self.R_.array[idx]
        X_next = self.X_next_.array[idx]
        return X, A, R, X_next

    def pop(self):
        """
        Pop the newest cached transition.

        Returns
        -------
        X, A, R, X_next : arrays
            A batch of preprocessed transitions. The size of the first axis of
            each of the four arrays matches and are equal to `batch_size=1`.

        """
        self._check_fitted()
        X = np.expand_dims(self.X_.pop(), axis=0)
        A = np.expand_dims(self.A_.pop(), axis=0)
        R = np.expand_dims(self.R_.pop(), axis=0)
        X_next = np.expand_dims(self.X_next_.pop(), axis=0)
        return X, A, R, X_next

    def popleft(self):
        """
        Pop the oldest cached transition.

        Returns
        -------
        X, A, R, X_next : arrays
            A batch of preprocessed transitions. The size of the first axis of
            each of the four arrays matches and are equal to `batch_size=1`.

        """
        self._check_fitted()
        X = np.expand_dims(self.X_.popleft(), axis=0)
        A = np.expand_dims(self.A_.popleft(), axis=0)
        R = np.expand_dims(self.R_.popleft(), axis=0)
        X_next = np.expand_dims(self.X_next_.popleft(), axis=0)
        return X, A, R, X_next

    def popleft_nstep(self, n=1):
        """
        Pop the oldest cached transition and return the `R` and `X_next` that
        correspond to an n-step look-ahead.

        .. note::

            To understand of what's going in this method, have a look at
            chapter 7 of `Sutton & Barto
            <http://incompleteideas.net/book/the-book-2nd.html>`_.

        Parameters
        ----------
        n : int
            The number of steps in the n-step bootstrapping procedure.

        Returns
        -------
        X, A, R, X_next : arrays
            A batch of preprocessed transitions. `X` and `A` correspond to the
            to-be-updated timestep :math:`\\tau=t-n+1`, while `X_next`
            corresponds to the look-ahead timestep :math:`\\tau+n=t+1`. `R`
            contains all the observed rewards between timestep :math:`\\tau+1`
            and :math:`\\tau+n` (inclusive), i.e. `R` represents the sequence
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{\\tau+n})`. This sequence
            is truncated to a size smaller than :math:`n` as we approach the
            end of the episode, where :math:`t>T-n`. The sequence becomes
            :math:`(R_\\tau, R_{\\tau+1}, \\dots, R_{T})`. In this phase of the
            replay, we can longer do a bootstrapping type look-ahead, which
            means that `X_next=None` until the end of the episode.

        """
        self._check_fitted()
        X = np.expand_dims(self.X_.popleft(), axis=0)
        A = np.expand_dims(self.A_.popleft(), axis=0)

        R = self.R_.array[:n]
        self.R_.popleft()

        X_next = self.X_next_.array[[-1]] if len(self.X_next_) >= n else None
        self.X_next_.popleft()

        return X, A, R, X_next

    def _check_fitted(self, raise_error=True):
        fitted = all((
            hasattr(self, 'X_'),
            hasattr(self, 'A_'),
            hasattr(self, 'R_'),
            hasattr(self, 'X_next_')))
        if raise_error and not fitted:
            raise NoExperienceCacheError("no experience has yet been recorded")
        return fitted
