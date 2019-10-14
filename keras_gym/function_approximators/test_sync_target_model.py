from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, UP, DOWN, LEFT, RIGHT

import pytest
import numpy as np
import tensorflow as tf

from ..caching import MonteCarloCache
from ..wrappers import TrainMonitor
from .generic import FunctionApproximator
from .value_v import V
from .value_q import QTypeI, QTypeII
from .policy_categorical import SoftmaxPolicy


if tf.__version__ >= '2.0':
    tf.compat.v1.disable_eager_execution()  # otherwise incredibly slow


# the MDP
actions = {LEFT: 'L', RIGHT: 'R', UP: 'U', DOWN: 'D'}
env = FrozenLakeEnv(is_slippery=False)
env = TrainMonitor(env)


# define function approximator
func = FunctionApproximator(env, lr=0.01)


def test_v():
    v = V(func, gamma=0.9, bootstrap_n=1)

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            v.update(s, r, done)
            s = s_next

    expected = v.predict_model.get_weights()[0]
    actual = v.target_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    v.sync_target_model()
    actual = v.target_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_q1():
    q = QTypeI(func, gamma=0.9, bootstrap_n=1, update_strategy='sarsa')

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            q.update(s, a, r, done)
            s = s_next

    expected = q.predict_model.get_weights()[0]
    actual = q.target_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    q.sync_target_model()
    actual = q.target_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_q2():
    q = QTypeII(func, gamma=0.9, bootstrap_n=1, update_strategy='sarsa')

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            q.update(s, a, r, done)
            s = s_next

    expected = q.predict_model.get_weights()[0]
    actual = q.target_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    q.sync_target_model()
    actual = q.target_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_pi_vanilla():
    pi = SoftmaxPolicy(func, update_strategy='vanilla')
    cache = MonteCarloCache(env, gamma=0.9)

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            cache.add(s, a, r, done)
            if done:
                while cache:
                    S, A, G = cache.pop()
                    pi.batch_update(S, A, G)
            s = s_next

    expected = pi.predict_model.get_weights()[0]
    actual = pi.target_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    pi.sync_target_model()
    actual = pi.target_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_pi_ppo():
    pi = SoftmaxPolicy(func, update_strategy='ppo')
    cache = MonteCarloCache(env, gamma=0.9)

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            cache.add(s, a, r, done)
            if done:
                while cache:
                    S, A, G = cache.pop()
                    pi.batch_update(S, A, G)
            s = s_next

    expected = pi.predict_model.get_weights()[0]
    actual = pi.target_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    pi.sync_target_model()
    actual = pi.target_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_pi_param_ppo():
    pi = SoftmaxPolicy(func, update_strategy='ppo')
    cache = MonteCarloCache(env, gamma=0.9)

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            cache.add(s, a, r, done)
            if done:
                while cache:
                    S, A, G = cache.pop()
                    pi.batch_update(S, A, G)
            s = s_next

    expected = pi.predict_param_model.get_weights()[0]
    actual = pi.target_param_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    pi.sync_target_model()
    actual = pi.target_param_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)


def test_pi_greedy_ppo():
    pi = SoftmaxPolicy(func, update_strategy='ppo')
    cache = MonteCarloCache(env, gamma=0.9)

    # run episodes
    for _ in range(20):
        s = env.reset()
        for a in [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT]:
            s_next, r, done, info = env.step(a)
            cache.add(s, a, r, done)
            if done:
                while cache:
                    S, A, G = cache.pop()
                    pi.batch_update(S, A, G)
            s = s_next

    expected = pi.predict_greedy_model.get_weights()[0]
    actual = pi.target_greedy_model.get_weights()[0]

    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        np.testing.assert_array_almost_equal(actual, expected)

    pi.sync_target_model()
    actual = pi.target_greedy_model.get_weights()[0]
    np.testing.assert_array_almost_equal(actual, expected)
