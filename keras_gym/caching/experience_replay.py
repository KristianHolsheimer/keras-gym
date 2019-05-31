import numpy as np

from ..base.mixins import RandomStateMixin
from ..base.errors import NumpyArrayCheckError, InsufficientCacheError
from ..utils import check_numpy_array, get_env_attr


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

    random_seed : int or None

        To get reproducible results.


    """
    def __init__(
            self,
            capacity,
            batch_size=32,
            num_frames=None,
            bootstrap_n=1,
            gamma=0.99,
            random_seed=None):

        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.num_frames = int(num_frames or 1)
        self.bootstrap_n = int(bootstrap_n)
        self.gamma = float(gamma)
        self.random_seed = random_seed

        # internal
        self._initialized = False

    @classmethod
    def from_value_function(cls, qfunction, capacity, batch_size=32):
        """
        Create a new instance by extracting some settings from a Q-function.

        The settings that are extracted from the Q-function are: ``gamma``,
        ``bootstrap_n`` and ``num_frames``. The latter is taken from the
        Q-function's ``env`` attribute.

        Parameters
        ----------
        qfunction : Q-function object

            A state-action value function.

        capacity : positive int

            The capacity of the experience replay buffer. DQN typically uses
            ``capacity=1000000``.

        batch_size : positive int, optional

            The desired batch size of the sample.

        Returns
        -------
        experience_replay_buffer

            A new instance.

        """
        self = cls(
            capacity=capacity,
            batch_size=batch_size,
            gamma=qfunction.gamma,
            bootstrap_n=qfunction.bootstrap_n,
            num_frames=get_env_attr(qfunction.env, 'num_frames'))
        return self

    def add(self, s, a, r, done, episode_id):
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

        done : bool

            Whether the episode has finished.

        episode_id : int

            The episode in which the transition took place. This is needed for
            generating consistent samples.

        """
        s = self._extract_last_frame(s)

        if not self._initialized:
            self._s_shape = s.shape
            self._s_dtype = s.dtype
            self._init_cache()

        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        self._d[self._i] = done
        self._e[self._i] = episode_id
        self._i = (self._i + 1) % (self.capacity + self.bootstrap_n)
        if self._num_transitions < self.capacity + self.bootstrap_n:
            self._num_transitions += 1

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

                \\left( R^{(n)}_t + I_t\\,Q(S_{t+n},A_{t+n})
                    - Q(S_t, A_t) \\right)^2

        """  # noqa: E501
        if not self._initialized or len(self) < self.batch_size:
            raise InsufficientCacheError(
                "insufficient cached data to sample from")

        S = []
        A = []
        Rn = []
        S_next = []
        A_next = []
        I_next = []

        for attempt in range(10 * self.batch_size):
            # js are the S indices and ks are the S_next indices
            J = len(self) - self.num_frames
            assert J > 0, "please insert more transitions before sampling"
            js = self.random.randint(J) + np.arange(self.num_frames)
            ks = js + self.bootstrap_n
            ls = np.arange(js[-1], ks[-1])

            # wrap around
            js %= self.capacity + self.bootstrap_n
            ks %= self.capacity + self.bootstrap_n
            ls %= self.capacity + self.bootstrap_n

            # check if S indices are all from the same episode
            ep = self._e[js[-1]]
            if any(self._e[j] not in (ep, ep - 1) for j in js[:-1]):
                # Check if all js are from the current episode or from the
                # immediately preceding episode. Otherwise, we would generate
                # spurious data because it would probably mean that 'js' spans
                # the overwrite-boundary.
                continue
            for i, j in reversed(list(enumerate(js[:-1]))):
                # if j is from previous episode, replace it by its successor
                if self._e[j] < ep:
                    js[i] = js[i + 1]

            # gather partial returns
            rn = np.zeros(1)
            done = False
            for t, l in enumerate(ls):
                rn[0] += pow(self.gamma, t) * self._r[l]
                done = self._d[l]
                if done:
                    break

            if not done and any(self._e[k] != ep for k in ks):
                continue

            # permutation to transpose 'num_frames' axis to axis=-1
            perm = np.roll(np.arange(self._s.ndim), -1)
            S.append(self._s[js].transpose(perm))
            A.append(self._a[js[-1:]])
            Rn.append(rn)
            S_next.append(self._s[ks].transpose(perm))
            A_next.append(self._a[ks[-1:]])
            if done:
                I_next.append(np.zeros(1))
            else:
                I_next.append(
                    np.power([self.gamma], self.bootstrap_n))

            if len(S) == self.batch_size:
                break

        if len(S) < self.batch_size:
            raise RuntimeError("couldn't construct valid sample")

        S = np.stack(S, axis=0)
        A = np.concatenate(A, axis=0)
        Rn = np.concatenate(Rn, axis=0)
        I_next = np.concatenate(I_next, axis=0)
        S_next = np.stack(S_next, axis=0)
        A_next = np.concatenate(A_next, axis=0)

        if self.num_frames == 1:
            S = np.squeeze(S, axis=-1)
            S_next = np.squeeze(S_next, axis=-1)

        return S, A, Rn, I_next, S_next, A_next

    def clear(self):
        """
        Clear the experience replay buffer.

        """
        self._i = 0
        self._num_transitions = 0

    def _init_cache(self):
        self._i = 0
        self._num_transitions = 0

        n = (self.capacity + self.bootstrap_n,)
        self._s = np.empty(n + self._s_shape, self._s_dtype)  # frame
        self._a = np.zeros(n, 'int32')      # actions taken (assume Discrete)
        self._r = np.zeros(n, 'float')      # rewards
        self._d = np.zeros(n, 'bool')       # done?
        self._e = np.zeros(n, 'int32')      # episode id
        self._initialized = True

    def _extract_last_frame(self, s):
        if self.num_frames == 1:
            return s

        check_numpy_array(s, axis_size=self.num_frames, axis=-1)
        if s.ndim == 3:
            s = s[:, :, -1]
        elif s.ndim == 4:
            s = s[:, :, :, -1]
        else:
            NumpyArrayCheckError(
                "expected ndim equal to 3 or 4, got shape: {}".format(s.shape))
        return s

    def __len__(self):
        return max(0, self._num_transitions - self.bootstrap_n)

    def __bool__(self):
        return bool(len(self))
