import numpy as np
import tensorflow as tf

from ..utils import softmax, idx
from ..losses import SoftmaxPolicyLossWithLogits


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
