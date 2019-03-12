import numpy as np
import tensorflow as tf
import pytest

from ..utils import softmax, idx
from ..losses import SoftmaxPolicyLossWithLogits, masked_mse_loss


class TestMaskedMseLoss:
    y_true = tf.placeholder(shape=[None, None], dtype=tf.float64)
    y_pred = tf.placeholder(shape=[None, None], dtype=tf.float64)

    # create feed_dict
    rnd = np.random.RandomState(7)
    batch_size = 5
    num_actions = 3
    y_true_numpy = rnd.randn(batch_size, num_actions)
    y_pred_numpy = y_true_numpy.copy()
    A = rnd.randint(num_actions, size=batch_size)
    y_pred_numpy[idx(A), A] = rnd.randn(batch_size)
    feed_dict = {y_true: y_true_numpy, y_pred: y_pred_numpy}

    def _reference_implementation(self):
        """ reference implementation in numpy """

        # inputs
        y_true = self.y_true_numpy
        y_pred = self.y_pred_numpy

        # residuals
        err = y_pred - y_true

        # bad implementation
        mse_unmasked = np.mean(err ** 2)

        # good implementation
        mask = (y_pred != y_true)
        mse_masked = np.mean(err[mask] ** 2)

        np.testing.assert_almost_equal(mask.mean(), 1.0 / self.num_actions)
        with pytest.raises(AssertionError):
            np.testing.assert_almost_equal(mse_masked, mse_unmasked)

        return mse_masked

    def test_masked_mse_loss(self):
        expected_loss = self._reference_implementation()

        # tensorflow computation graph
        loss_op = masked_mse_loss(self.y_true, self.y_pred)

        with tf.Session() as s:
            loss = s.run(loss_op, self.feed_dict)

        np.testing.assert_almost_equal(loss, expected_loss)


class TestSoftmaxPolicyLossWithLogits:
    A = tf.placeholder(shape=[None, None], dtype=tf.int64)
    logits = tf.placeholder(shape=[None, None], dtype=tf.float64)
    advantages = tf.placeholder(shape=[None], dtype=tf.float64)

    # create feed_dict
    rnd = np.random.RandomState(6)
    batch_size = 5
    num_actions = 3
    feed_dict = {
        A: rnd.randint(num_actions, size=(batch_size, 1)),
        logits: rnd.randn(batch_size, num_actions),
        advantages: rnd.randn(batch_size)}

    def _reference_implementation(self):
        """ reference implementation in numpy """

        # inputs
        A = self.feed_dict[self.A]             # a.k.a. y_true
        logits = self.feed_dict[self.logits]   # a.k.a. y_pred
        adv = self.feed_dict[self.advantages]

        # compute surrogate loss
        A = np.squeeze(A, axis=1)
        pi = softmax(logits, axis=1)
        mean_logits = np.einsum('ij,ij->i', logits, pi)
        logpi_all = logits - mean_logits[:, np.newaxis]
        logpi = logpi_all[idx(A), A]
        surrogate_loss = -np.mean(adv * logpi)
        return surrogate_loss

    def test_call(self):
        # expected output
        expected_loss = self._reference_implementation()

        # tensorflow computation graph
        loss_function = SoftmaxPolicyLossWithLogits(self.advantages)
        loss_op = loss_function(self.A, self.logits)

        with tf.Session() as s:
            loss = s.run(loss_op, self.feed_dict)

        np.testing.assert_almost_equal(loss, expected_loss)
