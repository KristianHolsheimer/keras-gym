import logging
import numpy as np
from scipy.special import expit as sigmoid, betaln, digamma
# import pytest

from tensorflow.keras import backend as K

from .misc import set_tf_loglevel
from .array import project_onto_actions_np, log_softmax
from .tensor import (
    project_onto_actions_tf, log_softmax_tf, diff_transform_matrix, entropy,
    cross_entropy)


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


def test_log():
    a = 1e6
    x = np.log(a).astype('float32')

    # compare
    a = K.constant(a, dtype='float32')
    y = K.get_session().run(K.log(a))

    np.testing.assert_almost_equal(x, y)


def test_logbeta_small_value():
    import tensorflow as tf
    ab = [1, 2]
    x = betaln(*ab).astype('float32')

    # compare
    ab = K.constant(ab, dtype='float32')
    y = K.get_session().run(tf.math.lbeta(ab))

    np.testing.assert_almost_equal(x, y)


def test_logbeta_batch():
    import tensorflow as tf
    ab = np.array([[11, 7], [1, 3], [4, 5]])
    a, b = ab[:, 0], ab[:, 1]
    x = betaln(a, b).astype('float32')

    # compare
    ab = K.constant(ab, dtype='float32')
    y = K.get_session().run(tf.math.lbeta(ab))

    np.testing.assert_almost_equal(x, y)


def test_entropy_beta_expected():
    rnd = np.random.RandomState(13)
    batch_size = 5
    actions_ndim = 3
    Z = rnd.randn(batch_size, actions_ndim, 2).astype('float32')
    Z[:, :, 0] = sigmoid(Z[:, :, 0])  # p
    Z[:, :, 1] = np.exp(Z[:, :, 1])   # n

    # numpy implementation
    p, n = Z[:, :, 0], Z[:, :, 1]
    p = np.maximum(1e-16, p)
    n = np.maximum(1e-16, n)
    assert np.sum(n >= 1e4) == 0
    assert n.shape == (batch_size, actions_ndim)
    assert p.shape == (batch_size, actions_ndim)
    a, b = n * p, n * (1 - p)
    H_np = betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)  # noqa: E501
    assert H_np.shape == (batch_size, actions_ndim)

    # compare
    Z = K.constant(Z, dtype='float32')
    H_tf = K.get_session().run(entropy(Z, 'beta'))
    assert H_tf.shape == H_np.shape
    np.testing.assert_array_almost_equal(H_tf, H_np, decimal=5)


def test_entropy_beta_clip_p():
    rnd = np.random.RandomState(13)
    batch_size = 5
    actions_ndim = 3
    Z = rnd.randn(batch_size, actions_ndim, 2).astype('float32')
    Z[:, :, 0] = sigmoid(Z[:, :, 0])  # p
    Z[:, :, 1] = np.exp(Z[:, :, 1])   # n
    Z[0, 0, 0] = 1e-17  # p too small

    # numpy implementation
    p, n = Z[:, :, 0], Z[:, :, 1]
    assert np.sum(p < 1e-16) == 1  # count of lower-bound value reached for p
    assert np.sum(n < 1e-16) == 0  # count of lower-bound value reached for n
    assert np.sum(n >= 1e4) == 0   # count of upper-bound value reached for n
    p = np.maximum(1e-16, p)
    n = np.maximum(1e-16, n)
    assert n.shape == (batch_size, actions_ndim)
    assert p.shape == (batch_size, actions_ndim)
    a, b = n * p, n * (1 - p)
    H_np = betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)  # noqa: E501
    assert H_np.shape == (batch_size, actions_ndim)

    # compare
    Z = K.constant(Z, dtype='float32')
    H_tf = K.get_session().run(entropy(Z, 'beta'))
    assert H_tf.shape == H_np.shape
    np.testing.assert_array_almost_equal(H_tf, H_np, decimal=5)


def test_entropy_beta_clip_n():
    rnd = np.random.RandomState(13)
    batch_size = 5
    actions_ndim = 3
    Z = rnd.randn(batch_size, actions_ndim, 2).astype('float32')
    Z[:, :, 0] = sigmoid(Z[:, :, 0])  # p
    Z[:, :, 1] = np.exp(Z[:, :, 1])   # n
    Z[0, 0, 1] = 1e-17  # n too small

    # numpy implementation
    p, n = Z[:, :, 0], Z[:, :, 1]
    assert np.sum(p < 1e-16) == 0  # count of lower-bound value reached for p
    assert np.sum(n < 1e-16) == 1  # count of lower-bound value reached for n
    assert np.sum(n >= 1e4) == 0   # count of upper-bound value reached for n
    p = np.maximum(1e-16, p)
    n = np.maximum(1e-16, n)
    assert n.shape == (batch_size, actions_ndim)
    assert p.shape == (batch_size, actions_ndim)
    a, b = n * p, n * (1 - p)
    H_np = betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)  # noqa: E501
    assert H_np.shape == (batch_size, actions_ndim)

    # compare
    Z = K.constant(Z, dtype='float32')
    H_tf = K.get_session().run(entropy(Z, 'beta'))
    assert H_tf.shape == H_np.shape
    np.testing.assert_array_almost_equal(H_tf, H_np, decimal=5)


def test_cross_entropy_beta_expected():
    rnd = np.random.RandomState(13)
    batch_size = 5
    actions_ndim = 3
    P = rnd.randn(batch_size, actions_ndim, 2).astype('float32')
    P[:, :, 0] = sigmoid(P[:, :, 0])  # p
    P[:, :, 1] = np.exp(P[:, :, 1])   # n
    Z = rnd.randn(batch_size, actions_ndim, 2).astype('float32')
    Z[:, :, 0] = sigmoid(Z[:, :, 0])  # p
    Z[:, :, 1] = np.exp(Z[:, :, 1])   # n

    # numpy implementation
    p, n = P[:, :, 0], P[:, :, 1]
    p = np.maximum(1e-16, p)
    n = np.maximum(1e-16, n)
    p_, n_ = Z[:, :, 0], Z[:, :, 1]
    p_ = np.maximum(1e-16, p_)
    n_ = np.maximum(1e-16, n_)
    assert np.sum(n >= 1e4) == 0
    assert n.shape == (batch_size, actions_ndim)
    assert p.shape == (batch_size, actions_ndim)
    assert n_.shape == (batch_size, actions_ndim)
    assert p_.shape == (batch_size, actions_ndim)
    a, b = n * p, n * (1 - p)
    a_, b_ = n_ * p_, n_ * (1 - p_)
    H_np = betaln(a_, b_) - (a_ - 1) * digamma(a) - (b_ - 1) * digamma(b) + (a_ + b_ - 2) * digamma(a + b)  # noqa: E501
    assert H_np.shape == (batch_size, actions_ndim)

    # compare
    Z = K.constant(Z, dtype='float32')
    P = K.constant(P, dtype='float32')
    H_tf = K.get_session().run(cross_entropy(P, Z, 'beta'))
    assert H_tf.shape == H_np.shape
    np.testing.assert_array_almost_equal(H_tf, H_np, decimal=5)
