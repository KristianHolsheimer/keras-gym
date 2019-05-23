import numpy as np
import pytest

from tensorflow.keras import backend as K

from ..base.errors import NumpyArrayCheckError
from .helpers import (
    idx, check_numpy_array, project_onto_actions_np, project_onto_actions_tf,
    softmax)


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
    with K.get_session() as s:
        Y_proj_tf = s.run(Y_proj_tf)

    # compare
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
