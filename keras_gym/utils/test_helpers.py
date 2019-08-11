import numpy as np
import pytest

from tensorflow.keras import backend as K

from ..base.errors import NumpyArrayCheckError
from .helpers import (
    idx, check_numpy_array, project_onto_actions_np, project_onto_actions_tf,
    softmax, log_softmax, log_softmax_tf, diff_transform_matrix)


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


def test_project_onto_actions_tf_expected():
    # numpy
    Y_np = np.resize(np.arange(12), (3, 4))
    A_np = np.array([2, 0, 3])
    Y_proj_np = project_onto_actions_np(Y_np, A_np)

    # tensorflow
    Y_tf = K.constant(Y_np, dtype='int64')
    A_tf = K.constant(A_np, dtype='int64')
    Y_proj_tf = project_onto_actions_tf(Y_tf, A_tf)

    # compare
    Y_proj_tf = K.get_session().run(Y_proj_tf)
    np.testing.assert_array_equal(Y_proj_tf, Y_proj_np)


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


def test_log_softmax_tf_expected():
    rnd = np.random.RandomState(7)
    w_np = rnd.randn(3, 5).astype('float32')
    logx_np = log_softmax(w_np, axis=1)
    logy_np = log_softmax(w_np + 100., axis=1)
    logz_np = log_softmax(w_np * 100., axis=1)

    w_tf = K.constant(w_np)
    logx_tf = log_softmax_tf(w_tf, axis=1)
    logy_tf = log_softmax_tf(w_tf + 100., axis=1)
    logz_tf = log_softmax_tf(w_tf * 100., axis=1)

    logx_tf = K.get_session().run(logx_tf)
    logy_tf = K.get_session().run(logy_tf)
    logz_tf = K.get_session().run(logz_tf)

    np.testing.assert_array_almost_equal(logx_tf, logx_np)
    np.testing.assert_array_almost_equal(logy_tf, logy_np)
    np.testing.assert_array_almost_equal(logz_tf, logz_np)


def test_diff_transform_matrix():
    m1 = np.array([[1.0]])
    m2 = np.array([
        [-1, 0],
        [1, 1],
    ])
    m3 = np.array([
        [1, 0, 0],
        [-2, -1, 0],
        [1, 1, 1],
    ])
    m4 = np.array([
        [-1, 0, 0, 0],
        [3, 1, 0, 0],
        [-3, -2, -1, 0],
        [1, 1, 1, 1],
    ])

    # tests
    np.testing.assert_array_equal(
        m1, K.get_session().run(diff_transform_matrix(1)))
    np.testing.assert_array_equal(
        m2, K.get_session().run(diff_transform_matrix(2)))
    np.testing.assert_array_equal(
        m3, K.get_session().run(diff_transform_matrix(3)))
    np.testing.assert_array_equal(
        m4, K.get_session().run(diff_transform_matrix(4)))
