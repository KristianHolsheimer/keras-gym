import logging
import numpy as np
# import pytest

from tensorflow.keras import backend as K

from .misc import set_tf_loglevel
from .array import project_onto_actions_np, log_softmax
from .tensor import (
    project_onto_actions_tf, log_softmax_tf, diff_transform_matrix)


set_tf_loglevel(logging.ERROR)


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
