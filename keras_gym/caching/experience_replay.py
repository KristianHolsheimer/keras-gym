import numpy as np

from ..base.mixins import RandomStateMixin
from ..utils import Transition

__all__ = (
    'ExperienceReplayBuffer',
)


class ExperienceReplayBuffer(RandomStateMixin):
    """
    A simple numpy implementation of an experience replay buffer. This is
    written primarily with computer game environments (Atari) in mind.

    It implements a generic experience replay buffer for environments in which
    individual observations (frames) are stacked to represent the state.

    Parameters
    ----------
    state_preprocessor : function

        Function to use for preprocessing a single state observation (e.g. an
        individual frame).

    capacity : positive int

        The capacity of the experience replay buffer. DQN typically uses
        ``capacity=1000000``.

    batch_size : positive int, optional

        The desired batch size of the sample.

    num_frames : positive int

        The number of frames to stack together to make up one state input
        ``S``.

    bootstrap_n : positive int

        The number of steps over which to delay bootstrapping, i.e. n-step
        bootstrapping.

    gamma : float between 0 and 1

        Reward discount factor.

    warmup_period : positive int

        Number of transitions to collect before sampling.

    random_seed : int or None

        To get reproducible results.


    """
    def __init__(
            self,
            capacity=1000000,
            batch_size=32,
            num_frames=4,
            bootstrap_n=1,
            gamma=0.99,
            warmup_period=50000,
            random_seed=None):

        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.num_frames = int(num_frames)
        self.bootstrap_n = int(bootstrap_n)
        self.gamma = float(gamma)
        self.warmup_period = int(warmup_period)
        self.random_seed = random_seed

        if self.warmup_period <= self.bootstrap_n + self.num_frames:
            raise ValueError(
                "warmup_period should be much larger than "
                "(bootstrap_n + num_frames)")

        # internal
        self._i = 0
        self._len = 0
        self._initialized = False

    def add(self, s, a, r, gamma, episode_id):
        """
        Add a transition to the experience replay buffer.

        Parameters
        ----------
        s : state

            A single state observation.

        a : action

            A single action that generated this transition.

        r : float

            The observed rewards associated with this transition.

        gamma : bool

            By how much we want to bootstrap (can be zero).

        episode_id : int

            The episode in which the transition took place. This is needed for
            generating consistent samples.

        """
        if not self._initialized:
            self._init_cache(s)

        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        self._d[self._i] = gamma
        self._e[self._i] = episode_id
        self._i = (self._i + 1) % (self.capacity + 1)
        if self._len < self.capacity:
            self._len += 1

    def sample(self):
        """
        Get a batch of transitions to be used for bootstrapped updates.

        Returns
        -------
        S, A, Rn, I_next, S_next, A_next : tuple of arrays

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`A`, :term:`Rn`, :term:`I_next`, :term:`S_next`, :term:`A_next`)

            These are typically used for bootstrapped updates, e.g. minimizing
            the bootstrapped MSE:

            .. math::

                \\left( R^{(n)}_t + I_t\,Q(S_{t+n},A_{t+n})
                    - Q(S_t, A_t) \\right)^2

        """  # noqa: E501
        sample = {'S': [], 'A': [], 'Rn': [],
                  'S_next': [], 'A_next': [], 'I_next': []}

        for attempt in range(10 * self.batch_size):
            # js are the S indices and ks are the S_next indices
            J = len(self) - self.bootstrap_n - self.num_frames
            assert J > 0, "please insert more transitions before sampling"
            js = self._random.randint(J) + np.arange(self.num_frames)
            ks = js + self.bootstrap_n
            ls = np.arange(js[-1], ks[-1])

            # wrap around
            js %= self.capacity + 1
            ks %= self.capacity + 1
            ls %= self.capacity + 1

            # check if S indices are all from the same episode
            ep = self._e[js[0]]
            if any(self._e[j] != ep for j in js[1:]):
                continue

            # gather partial returns
            Rn = np.zeros(1)
            done = False
            for t, l in enumerate(ls):
                Rn[0] += pow(self.gamma, t) * self._r[l]
                done = self._d[l]
                if done:
                    break

            if not done and any(self._e[k] != ep for k in ks):
                continue

            # permutation to transpose 'num_frames' axis to axis=-1
            perm = np.roll(np.arange(self._s.ndim), -1)
            sample['S'].append(self._s[js].transpose(perm).ravel())
            sample['A'].append(self._a[js[-1:]])
            sample['Rn'].append(Rn)
            sample['S_next'].append(self._s[ks].transpose(perm).ravel())
            sample['A_next'].append(self._a[ks[-1:]])
            if done:
                sample['I_next'].append(np.zeros(1))
            else:
                sample['I_next'].append(
                    np.power([self.gamma], self.bootstrap_n))

            if len(sample['S']) == self.batch_size:
                break

        if len(sample['S']) < self.batch_size:
            raise RuntimeError("couldn't construct valid sample")

        S = np.stack(sample['S'])
        A = np.stack(sample['A'])
        Rn = np.stack(sample['Rn'])
        I_next = np.stack(sample['I_next'])
        S_next = np.stack(sample['S_next'])
        A_next = np.stack(sample['A_next'])

        return S, A, Rn, I_next, S_next, A_next

    def is_warming_up(self):
        return len(self) < self.warmup_period

    def _init_cache(self, x):
        n = (self.capacity + 1,)
        self._s = np.zeros(n + x.shape, x.dtype)  # frame
        self._a = np.zeros(n, 'int32')      # actions taken (assume Discrete)
        self._r = np.zeros(n, 'float')      # rewards
        self._d = np.zeros(n, 'bool')       # done?
        self._e = np.zeros(n, 'int32')      # episode id
        self._e[...] = -1
        self._initialized = True

    def __len__(self):
        return self._len

    def __bool__(self):
        return bool(len(self))
