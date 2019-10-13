import gym
import logging
import numpy as np
# import pytest

from tensorflow import keras
from tensorflow.keras import backend as K

from .misc import set_tf_loglevel
from .array import project_onto_actions_np, log_softmax
from .tensor import (
    project_onto_actions_tf, log_softmax_tf, diff_transform_matrix,
    reals_to_box_tf, box_to_reals_tf)


set_tf_loglevel(logging.ERROR)


def test_project_onto_actions_tf_expected():
    # numpy
    Y_np = np.resize(np.arange(12), (3, 4))
    A_np = np.array([2, 0, 3])
    expected = project_onto_actions_np(Y_np, A_np)

    # tensorflow
    Y = keras.Input(Y_np.shape[1:], dtype='int64')
    A = keras.Input(A_np.shape[1:], dtype='int64')
    Y_proj = project_onto_actions_tf(Y, A)

    # compare
    actual = keras.Model([Y, A], Y_proj).predict([Y_np, A_np])
    np.testing.assert_array_equal(actual, expected)


def test_log_softmax_tf_expected():
    rnd = np.random.RandomState(7)
    w_np = rnd.randn(3, 5).astype('float32')
    logx_expected = log_softmax(w_np, axis=1)
    logy_expected = log_softmax(w_np + 100., axis=1)
    logz_expected = log_softmax(w_np * 100., axis=1)

    w = keras.Input(w_np.shape[1:], dtype='float32')
    logx = log_softmax_tf(w, axis=1)
    logy = log_softmax_tf(w + 100., axis=1)
    logz = log_softmax_tf(w * 100., axis=1)

    logx_actual = keras.Model(w, logx).predict(w_np)
    logy_actual = keras.Model(w, logy).predict(w_np)
    logz_actual = keras.Model(w, logz).predict(w_np)

    np.testing.assert_array_almost_equal(logx_actual, logx_expected)
    np.testing.assert_array_almost_equal(logy_actual, logy_expected)
    np.testing.assert_array_almost_equal(logz_actual, logz_expected, decimal=5)


def test_diff_transform_matrix():
    m1_expected = np.array([[1.0]])
    m2_expected = np.array([
        [-1, 0],
        [1, 1],
    ])
    m3_expected = np.array([
        [1, 0, 0],
        [-2, -1, 0],
        [1, 1, 1],
    ])
    m4_expected = np.array([
        [-1, 0, 0, 0],
        [3, 1, 0, 0],
        [-3, -2, -1, 0],
        [1, 1, 1, 1],
    ])

    i = keras.Input([])
    m1 = keras.layers.Lambda(
        lambda x: K.expand_dims(diff_transform_matrix(1), axis=0))(i)
    m2 = keras.layers.Lambda(
        lambda x: K.expand_dims(diff_transform_matrix(2), axis=0))(i)
    m3 = keras.layers.Lambda(
        lambda x: K.expand_dims(diff_transform_matrix(3), axis=0))(i)
    m4 = keras.layers.Lambda(
        lambda x: K.expand_dims(diff_transform_matrix(4), axis=0))(i)

    m1_actual = keras.Model(i, m1).predict([42])[0]
    m2_actual = keras.Model(i, m2).predict([42])[0]
    m3_actual = keras.Model(i, m3).predict([42])[0]
    m4_actual = keras.Model(i, m4).predict([42])[0]

    # tests
    np.testing.assert_array_equal(m1_actual, m1_expected)
    np.testing.assert_array_equal(m2_actual, m2_expected)
    np.testing.assert_array_equal(m3_actual, m3_expected)
    np.testing.assert_array_equal(m4_actual, m4_expected)


def test_box_to_reals_tf():
    rnd = np.random.RandomState(13)
    space = gym.spaces.Box(-rnd.rand(3), rnd.rand(3))
    space.np_random = rnd

    A_expected = np.vstack((
        space.sample(),
        space.sample(),
    ))

    A = keras.Input(A_expected.shape[1:])
    A_actual = keras.layers.Lambda(
        lambda x: reals_to_box_tf(box_to_reals_tf(x, space), space))(A)
    A_actual = keras.Model(A, A_actual).predict(A_expected)
    np.testing.assert_array_almost_equal(A_actual, A_expected)
