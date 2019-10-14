import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from scipy.stats import multinomial

from ..utils.array import one_hot

from .categorical import CategoricalDist

if tf.__version__ >= '2.0':
    tf.random.set_seed(11)
else:
    tf.set_random_seed(11)
rnd = np.random.RandomState(13)

x_np = rnd.randn(7, 5)
y_np = rnd.randint(3, size=7)
y_onehot_np = one_hot(y_np, 3)

x = keras.Input([5], dtype='float32')
y = keras.Input([1], dtype='int32')
y_onehot = keras.Input([3], dtype='float32')

logits = keras.layers.Dense(3)(x)
proba = keras.layers.Lambda(K.softmax)(logits)

dist = CategoricalDist(logits)
sample = keras.layers.Lambda(lambda args: dist.sample())(logits)

# test against scipy implementation
proba_np = keras.Model(x, proba).predict(x_np)
dists_np = [multinomial(n=1, p=p) for p in proba_np]  # cannot broadcast


def test_sample():
    # if tf.__version__ >= '2.0':
    #     expected = one_hot(np.array([1, 1, 1, 2, 1, 0, 2]), n=3)
    # else:
    #     expected = one_hot(np.array([1, 1, 1, 2, 1, 0, 2]), n=3)

    actual = keras.Model(x, sample).predict(x_np)

    assert actual.shape == (7, 3)
    np.testing.assert_array_almost_equal(actual.sum(axis=1), np.ones(7))
    # np.testing.assert_array_almost_equal(actual, expected)


def test_log_proba():
    expected = np.stack([d.logpmf(a) for d, a in zip(dists_np, y_onehot_np)])

    out = keras.layers.Lambda(lambda args: dist.log_proba(y))(logits)
    actual = keras.Model([x, y], out).predict([x_np, y_np])

    np.testing.assert_array_almost_equal(actual, expected)


def test_log_proba_onehot():
    expected = np.stack([d.logpmf(a) for d, a in zip(dists_np, y_onehot_np)])

    out = keras.layers.Lambda(lambda args: dist.log_proba(y_onehot))(logits)
    actual = keras.Model([x, y_onehot], out).predict([x_np, y_onehot_np])

    np.testing.assert_array_almost_equal(actual, expected)


def test_entropy():
    expected = np.stack([d.entropy() for d in dists_np])

    out = keras.layers.Lambda(lambda args: dist.entropy())(logits)
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
