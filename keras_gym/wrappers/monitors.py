import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ..base.mixins import ActionSpaceMixin, LoggerMixin


__all__ = (
    'TrainMonitor',
)


class TrainMonitor(gym.Wrapper, ActionSpaceMixin, LoggerMixin):
    """
    Environment wrapper for monitoring the training process.

    This wrapper logs some diagnostics at the end of each episode and it also
    gives us some handy attributes (listed below).

    Parameters
    ----------
    env : gym environment

        A gym environment.

    tensorboard_dir : str, optional

        If provided, TrainMonitor will log all diagnostics to be viewed in
        tensorboard. To view these, point tensorboard to the same dir:

        .. code::

            $ tensorboard --logdir {tensorboard_dir}

    Attributes
    ----------
    T : positive int

        Global step counter. This is not reset by ``env.reset()``, use
        ``env.reset_global()`` instead.

    ep : positive int

        Global episode counter. This is not reset by ``env.reset()``, use
        ``env.reset_global()`` instead.

    t : positive int

        Step counter within an episode.

    G : float

        The return, i.e. amount of reward accumulated from the start of the
        current episode.

    avg_G : float

        The average return G, averaged over the past 100 episodes.

    dt_ms : float

        The average wall time of a single step, in milliseconds.

    """
    def __init__(self, env, tensorboard_dir=None):
        super().__init__(env)
        self.quiet = False
        self.reset_global()

        self.tensorboard = None
        if tensorboard_dir is not None:
            self.tensorboard = tf.summary.FileWriter(
                tensorboard_dir, flush_secs=10)

    def reset_global(self):
        """ Reset the global counters, not just the episodic ones. """
        self.T = 0
        self.ep = 0
        self.t = 0
        self.G = 0.0
        self.avg_G = 0.0
        self._n_avg_G = 0.0
        self._ep_starttime = time.time()
        self._ep_losses = None
        self._ep_actions = []
        self._losses = None

    def reset(self):
        # increment global counters:
        self.T += 1
        self.ep += 1
        # reset episodic counters:
        self.t = 0
        self.G = 0.0
        self._ep_starttime = time.time()
        self._ep_losses = None
        self._ep_actions = []
        return self.env.reset()

    @property
    def dt_ms(self):
        if self.t <= 0:
            return np.nan
        return 1000 * (time.time() - self._ep_starttime) / self.t

    @property
    def avg_r(self):
        if self.t <= 0:
            return np.nan
        return self.G / self.t

    def step(self, a):
        self._ep_actions.append(a)
        s_next, r, done, info = self.env.step(a)
        if info is None:
            info = {}
        info['monitor'] = {'T': self.T, 'ep': self.ep}
        self.t += 1
        self.T += 1
        self.G += r
        if done:
            if self._n_avg_G < 100:
                self._n_avg_G += 1.
            self.avg_G += (self.G - self.avg_G) / self._n_avg_G
            if not self.quiet:
                self.logger.info(
                    "ep: {:d}, T: {:,d}, G: {:.3g}, avg_G: {:.3g}, t: {:d}, "
                    "dt: {:.3f}ms{:s}"
                    .format(
                        self.ep, self.T, self.G, self.avg_G, self.t,
                        self.dt_ms, self._losses_str()))
            if self.tensorboard is not None:
                diagnostics = {
                    'ep_return': self.G, 'ep_avg_reward': self.avg_r,
                    'ep_steps': self.t, 'avg_step_duration_ms': self.dt_ms}
                if self._ep_losses is not None:
                    diagnostics.update(self._ep_losses)
                self._write_scalars_to_tensorboard(diagnostics)
                self._write_histogram_to_tensorboard(
                    values=self._ep_actions, name='actions',
                    is_discrete=self.action_space_is_discrete)
                self.tensorboard.flush()

        return s_next, r, done, info

    def _write_scalars_to_tensorboard(self, diagnostics):

        for k, v in diagnostics.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(
                    tag=f'TrainMonitor/{k}', simple_value=v)])
            self.tensorboard.add_summary(summary, global_step=self.T)

    def _write_histogram_to_tensorboard(self, values, name, is_discrete, bins=50):  # noqa: E501
        """
        This custom histogram logger was taken from:

            https://stackoverflow.com/a/48876774/2123555

        """

        if is_discrete:
            values = np.array(values, dtype='int')
            distinct_values, counts = np.unique(values, return_counts=True)
            bin_edges = distinct_values + 1
        else:
            values = np.array(values, dtype='float')
            counts, bin_edges = np.histogram(values, bins=bins)
            bin_edges = bin_edges[1:]

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=f'TrainMonitor/{name}', histo=hist)])
        self.tensorboard.add_summary(summary, global_step=self.T)

    def record_losses(self, losses):
        """
        Record losses during the training process.

        These are used to print more diagnostics.

        Parameters
        ----------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        if self._losses is None or set(self._losses) != set(losses):
            self._losses = dict(losses)
            self._n_losses = 1.0
        else:
            if self._n_losses < 100:
                self._n_losses += 1.0
            self._losses = {
                k: v + (losses[k] - v) / self._n_losses
                for k, v in self._losses.items()}

        if self._ep_losses is None or set(self._ep_losses) != set(losses):
            self._ep_losses = dict(losses)
            self._n_ep_losses = 1.0
        else:
            self._n_ep_losses += 1.0
            self._ep_losses = {
                k: v + (losses[k] - v) / self._n_ep_losses
                for k, v in self._ep_losses.items()}

    def _losses_str(self):
        if self._losses is not None:
            return ", " + ", ".join(
                '{:s}: {:.3g}'.format(k, v) for k, v in self._losses.items())
        return ""
