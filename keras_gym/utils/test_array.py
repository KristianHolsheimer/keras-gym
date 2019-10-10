import logging
import numpy as np
import gym
import pytest

from ..base.errors import NumpyArrayCheckError
from .misc import set_tf_loglevel
from .array import (
    idx, check_numpy_array, project_onto_actions_np, softmax, log_softmax,
    box_to_reals_np, reals_to_box_np)


set_tf_loglevel(logging.ERROR)


def test_check_numpy_array_ndim_min():
    arr = np.array([])
    with pytest.raises(NumpyArrayCheckError):
        check_numpy_array(arr, ndim_min=2)


def test_idx_type():
    with pytest.raises(NumpyArrayCheckError):
        idx(0)
    with pytest.raises(NumpyArrayCheckError):
        idx('foo')
    with pytest.raises(NumpyArrayCheckError):
        idx([])
    with pytest.raises(NumpyArrayCheckError):
        idx(None)


def test_idx_empty():
    arr = np.array(0)
    with pytest.raises(NumpyArrayCheckError):
        idx(arr)


def test_idx_expected():
    arr = np.resize(np.arange(12), (3, 4))
    np.testing.assert_array_equal(idx(arr), [0, 1, 2])
    assert idx(arr).dtype == 'int'


def test_project_onto_actions_np_expected():
    Y = np.resize(np.arange(12), (3, 4))
    A = np.array([2, 0, 3])
    Y_proj = project_onto_actions_np(Y, A)
    assert Y_proj.ndim == 1
    assert Y_proj.shape == (3,)
    np.testing.assert_array_equal(Y_proj, [Y[0, 2], Y[1, 0], Y[2, 3]])


def test_softmax_expected():
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


def test_log_softmax_expected():
    rnd = np.random.RandomState(7)
    w = rnd.randn(3, 5)
    x = softmax(w, axis=1)
    logx = log_softmax(w, axis=1)
    logy = log_softmax(w + 100., axis=1)
    logz = log_softmax(w * 100., axis=1)

    # check shape
    assert x.shape == w.shape

    # check against regular implementation
    np.testing.assert_almost_equal(np.log(x), logx)

    # check translation invariance
    np.testing.assert_almost_equal(logy, logx)

    # check robustness by clipping
    assert not np.any(np.isnan(logz))

    # check normalization for large values
    np.testing.assert_array_almost_equal(np.exp(logz).sum(axis=1), np.ones(3))


def test_box_to_reals_np():
    rnd = np.random.RandomState(13)
    space = gym.spaces.Box(-rnd.rand(3), rnd.rand(3))
    space.np_random = rnd

    A_expected = np.vstack((
        space.sample(),
        space.sample(),
    ))
    A_actual = reals_to_box_np(box_to_reals_np(A_expected, space), space)
    np.testing.assert_array_almost_equal(A_actual, A_expected)
