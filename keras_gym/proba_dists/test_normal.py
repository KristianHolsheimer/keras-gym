from .normal import NormalDist

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm

if tf.__version__ >= '2.0':
    tf.random.set_seed(11)
else:
    tf.set_random_seed(11)
rnd = np.random.RandomState(13)

x_np = rnd.randn(5, 3)
y_np = rnd.randn(5, 2)

x = keras.Input([3])
y = keras.Input([2])

mu = keras.layers.Dense(2, name='mu')(x)
logvar = keras.layers.Dense(2, name='logvar')(x)

dist = NormalDist(mu, logvar)
sample = keras.layers.Lambda(lambda args: dist.sample())([mu, logvar])

# test against scipy implementation
mu_np, logvar_np = keras.Model(x, [mu, logvar]).predict(x_np)
dist_np = norm(loc=mu_np, scale=np.exp(logvar_np / 2))


def test_sample():
    # expected = np.array(
    #     [[-1.64546160, 1.32016470],
    #      [-0.66928166, 0.50807550],
    #      [2.808926600, -0.4206512],
    #      [-2.26104740, 0.65707680],
    #      [1.820780800, -1.6517701]], dtype='float32')

    actual = keras.Model(x, sample).predict(x_np)

    assert actual.shape == (5, 2)
    # np.testing.assert_array_almost_equal(actual, expected)


def test_log_proba():
    expected = np.mean(dist_np.logpdf(y_np), axis=1)

    out = keras.layers.Lambda(lambda args: dist.log_proba(y))([mu, logvar, y])
    actual = keras.Model([x, y], out).predict([x_np, y_np])

    np.testing.assert_array_almost_equal(actual, expected)


def test_entropy():
    expected = np.mean(dist_np.entropy(), axis=1)

    out = keras.layers.Lambda(lambda args: dist.entropy())([mu, logvar])
    actual = keras.Model(x, out).predict(x_np)

    np.testing.assert_array_almost_equal(actual, expected)


def test_cross_entropy():
    # TODO: test this without implementing the same thing in numpy
    pass


def test_kl_divergence():
    # TODO: test this without implementing the same thing in numpy
    pass


def test_proba_ratio():
    # TODO: test this without implementing the same thing in numpy
    pass
