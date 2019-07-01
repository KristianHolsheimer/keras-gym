import numpy as np

from ..base.mixins import RandomStateMixin, NumActionsMixin
from ..base.errors import NumpyArrayCheckError, InsufficientCacheError
from ..utils import check_numpy_array, get_env_attr


__all__ = (
    'ExperienceReplayBuffer',
)


class ExperienceReplayBuffer(RandomStateMixin, NumActionsMixin):
    """
    A simple numpy implementation of an experience replay buffer. This is
    written primarily with computer game environments (Atari) in mind.

    It implements a generic experience replay buffer for environments in which
    individual observations (frames) are stacked to represent the state.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to infer the number of stacked
        frames ``num_frames`` as well as the number of actions ``num_actions``.

    capacity : positive int

        The capacity of the experience replay buffer. DQN typically uses
        ``capacity=1000000``.

    batch_size : positive int, optional

        The desired batch size of the sample.

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
            env,
            capacity,
            batch_size=32,
            bootstrap_n=1,
            gamma=0.99,
            random_seed=None):

        self.env = env
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.num_frames = get_env_attr(env, 'num_frames', 1)
        self.bootstrap_n = int(bootstrap_n)
        self.gamma = float(gamma)
        self.random_seed = random_seed

        # internal
        self._initialized = False

    @classmethod
    def from_value_function(cls, value_function, capacity, batch_size=32):
        """
        Create a new instance by extracting some settings from a Q-function.

        The settings that are extracted from the value function are: ``gamma``,
        ``bootstrap_n`` and ``num_frames``. The latter is taken from the
        value function's ``env`` attribute.

        Parameters
        ----------
        value_function : value-function object

            A state value function or a state-action value function.

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
            env=value_function.env,
            capacity=capacity,
            batch_size=batch_size,
            gamma=value_function.gamma,
            bootstrap_n=value_function.bootstrap_n)
        return self

    def add(self, s, pi, r, done, episode_id):
        """
        Add a transition to the experience replay buffer.

        Parameters
        ----------
        s : state

            A single state observation.

        pi : int or 1d array, shape: [num_actions]

            Vector of action propensities under the behavior policy. This may
            be just an indicator if the action propensities are inferred
            through sampling. For instance, let's say our action space is
            :class:`Discrete(4)`, then passing ``pi = 2`` is equivalent to
            passing ``pi = [0, 0, 1, 0]``. Both would indicate that the action
            :math:`a=2` was drawn from the behavior policy.

        r : float

            The observed rewards associated with this transition.

        done : bool

            Whether the episode has finished.

        episode_id : int

            The episode in which the transition took place. This is needed for
            generating consistent samples.

        """
        s = self._extract_last_frame(s)
        pi = self.check_pi(pi)

        if not self._initialized:
            self._s_shape = s.shape
            self._s_dtype = s.dtype
            self._init_cache()

        self._s[self._i] = s
        self._p[self._i] = pi
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
        S, P, Rn, In, S_next, P_next : tuple of arrays

            The returned tuple represents a batch of preprocessed transitions:

                (:term:`S`, :term:`P`, :term:`Rn`, :term:`In`, :term:`S_next`, :term:`P_next`)

            These are typically used for bootstrapped updates, e.g. minimizing
            the bootstrapped MSE:

            .. math::

                \\left(
                    R^{(n)}_t
                    + I^{(n)}_t\\,\\sum_aP(a|S_{t+n})\\,Q(S_{t+n},a)
                    - \\sum_aP(a|S_t)\\,Q(S_t,a) \\right)^2

        """  # noqa: E501
        if not self._initialized or len(self) < self.batch_size:
            raise InsufficientCacheError(
                "insufficient cached data to sample from")

        S = []
        P = []
        Rn = []
        In = []
        S_next = []
        P_next = []

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
            P.append(self._p[js[-1:]])
            Rn.append(rn)
            S_next.append(self._s[ks].transpose(perm))
            P_next.append(self._p[ks[-1:]])
            if done:
                In.append(np.zeros(1))
            else:
                In.append(
                    np.power([self.gamma], self.bootstrap_n))

            if len(S) == self.batch_size:
                break

        if len(S) < self.batch_size:
            raise RuntimeError("couldn't construct valid sample")

        S = np.stack(S, axis=0)
        P = np.concatenate(P, axis=0)
        Rn = np.concatenate(Rn, axis=0)
        In = np.concatenate(In, axis=0)
        S_next = np.stack(S_next, axis=0)
        P_next = np.concatenate(P_next, axis=0)

        if self.num_frames == 1:
            S = np.squeeze(S, axis=-1)
            S_next = np.squeeze(S_next, axis=-1)

        return S, P, Rn, In, S_next, P_next

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
        s = self._s_shape
        a = (self.num_actions,)
        self._s = np.empty(n + s, self._s_dtype)  # frames
        self._p = np.zeros(n + a, 'float')        # action propensities
        self._r = np.zeros(n, 'float')            # rewards
        self._d = np.zeros(n, 'bool')             # done?
        self._e = np.zeros(n, 'int32')            # episode id
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
