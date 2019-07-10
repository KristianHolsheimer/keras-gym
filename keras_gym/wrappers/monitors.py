import time

import gym
import numpy as np
import pandas as pd

from ..base.mixins import LoggerMixin


__all__ = (
    'TrainMonitor',
)


class TrainMonitor(gym.Wrapper, LoggerMixin):
    """
    Environment wrapper for monitoring the training process.

    This wrapper logs some diagnostics at the end of each episode and it also
    gives us some handy attributes (listed below).

    Parameters
    ----------
    env : gym environment

        A gym environment.

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

        The amount of reward accumulated from the start of the current episode.

    avg_r : float

        The average reward received from the start of the episode.

    dt_ms : float

        The average wall time of a single step, in milliseconds.

    """
    def __init__(self, env):
        super().__init__(env)
        self.quiet = False
        self.reset_global()

    def reset_global(self):
        """ Reset the global counters, not just the episodic ones. """
        self.T = 0
        self.ep = 0
        self.t = 0
        self.G = 0.0
        self._ep_starttime = time.time()

    def reset(self):
        # increment global counters:
        self.T += 1
        self.ep += 1
        # reset episodic counters:
        self.t = 0
        self.G = 0.0
        self._ep_starttime = time.time()
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
        s_next, r, done, info = self.env.step(a)
        if info is None:
            info = {}
        info['monitor'] = {'T': self.T, 'ep': self.ep}
        self.t += 1
        self.T += 1
        self.G += r
        if done and not self.quiet:
            self.logger.info(
                "ep: {:d}, T: {:,d}, G: {:.3g}, avg(r): {:.3f}, t: {:d}, "
                "dt: {:.3f}ms{:s}"
                .format(
                    self.ep, self.T, self.G, self.avg_r, self.t, self.dt_ms,
                    self._losses_str()))

        return s_next, r, done, info

    def record_losses(self, losses):
        """
        Record losses during the training process.

        These are used to print more diagnostics.

        Parameters
        ----------
        losses : dict

            A dict of losses/metrics, of type ``{name <str>: value <float>}``.

        """
        if not hasattr(self, '_losses'):
            self._losses = pd.DataFrame(
                columns=list(losses.keys()),
                data=np.zeros((1000, len(losses)), dtype='float'))
            self._losses_i = 0
            self._losses_n = 0
        self._losses.iloc[self._losses_i] = pd.Series(losses)
        self._losses_i += 1
        self._losses_i %= 1000
        if self._losses_n < 1000:
            self._losses_n += 1

    def _losses_str(self):
        if hasattr(self, '_losses'):
            losses = self._losses.iloc[:self._losses_n].mean()
            return ", " + ", ".join(
                '{:s}: {:.3g}'.format(k, v) for k, v in losses.items())
        return ""
