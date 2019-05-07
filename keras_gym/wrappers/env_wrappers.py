from copy import deepcopy
from collections import deque

import gym
import numpy as np

from ..utils import Transition, ArrayDeque


class TransitionWrapper(gym.Wrapper):
    def __init__(
            self, env,
            bootstrap_n=1,
            gamma=0.9,
            state_preprocessor=None,
            state_action_preprocessor=None):

        super().__init__(env)

        # private
        self._bootstrap_n = bootstrap_n
        self._gamma = gamma
        self._gammas = np.power(gamma, np.arange(bootstrap_n))
        self._gamman = np.power(gamma, bootstrap_n)
        self._state_preprocessor = state_preprocessor
        self._state_action_preprocessor = state_action_preprocessor
        self._check_preprocessor()
        self._XAI_deque = deque([], maxlen=(bootstrap_n + 1))  # (X, A, I_next)
        self._R_deque = ArrayDeque(shape=(), maxlen=(bootstrap_n + 1))

        # public
        self.transitions = None

    def reset(self):
        self.transitions = deque([])
        self._s_prev = self.env.reset()
        return self._s_prev

    def step(self, a):
        if self.transitions is None:
            raise RuntimeError("must reset() env before running step()")
        s = self._s_prev
        s_next, r, done, info = self.env.step(a)
        X = self._preprocess(s, a)
        A = np.array([a])
        R = np.array([r])
        I_next = np.array([0. if done else self._gamman])
        if not isinstance(info, dict):
            info = {}

        self._XAI_deque.append((X, A, I_next))
        self._R_deque.append(R)

        if not done:
            # do nothing if bootstrap window is still being populated
            if len(self._R_deque) < self._bootstrap_n + 1:
                info['transition'] = None  # this is a "no-op"
                return s_next, r, done, info

            # otherwise, just pop one transition out of the deques
            assert len(self._R_deque) == self._bootstrap_n + 1
            G = np.array([self._gammas.dot(self._R_deque.array[:-1])])
            self._R_deque.popleft()
            X, A, I_next = self._XAI_deque.popleft()
            X_next, A_next, _ = self._XAI_deque[-1]
            self.transitions.append(
                Transition(X, A, G, X_next, A_next, I_next))
            return s_next, r, done, info

        # unroll remaining cache if episode is done
        X_next = np.zeros_like(X)   # dummy
        A_next = np.zeros_like(A)   # dummy
        I_next = np.zeros(1)        # no more bootstrapping
        remaining_transitions = []  # will collect transitions in reverse
        G = np.zeros(1)             # will accumulate returns

        while self._XAI_deque:
            G[0] += self._gamma * self._R_deque.pop()
            X, A, I_next = self._XAI_deque.pop()
            remaining_transitions.append(
                Transition(X, A, G, X_next, A_next, I_next))

        self.transitions.extend(reversed(remaining_transitions))
        return s_next, r, done, info

    def _check_preprocessor(self):
        phi_s = self._state_preprocessor
        phi_sa = self._state_action_preprocessor
        if callable(phi_s) and callable(phi_sa):
            raise TypeError(
                "please specify either state_preprocessor or "
                "state_action_preprocessor, not both")
        if not (callable(phi_s) or callable(phi_sa)):
            raise TypeError(
                "must specify either state_preprocessor or "
                "state_action_preprocessor")

    def _preprocess(self, s, a):
        if callable(self._state_preprocessor):
            return self._state_preprocessor(s)
        else:
            return self._state_action_preprocessor(s, a)


class PeriodicCounter:
    COUNTER_TYPES = ('step', 'episode')

    def __init__(
            self,
            name: str,
            period: int,
            counter_type: str = 'step',
            start_value: int = 0):

        if counter_type not in self.COUNTER_TYPES:
            raise ValueError(
                "Bad counter_type: '{}', expected one of: {}"
                .format(counter_type, self.COUNTER_TYPES))

        self.name = name
        self.period = period
        self.counter = start_value
        self.counter_type = counter_type

    def increment_and_check(self) -> bool:
        self.counter += 1
        check = self.counter >= self.period
        if check:
            self.counter = 0
        return check

    def __repr__(self) -> str:
        return (
            "PeriodicCounter<name='{name}', counter={counter}, "
            "period={period}, counter_type='{counter_type}'>"
            .format(**vars(self)))


class CounterWrapper(gym.Wrapper):
    """
    This environment wrapper keeps track of the number of episodes and
    the number of individual time steps that have elapsed since the start.

    Besides keeping track of these counters, it also allows for the addition of
    periodic step/episode counters. These are particularly useful for agents
    that depend on periodic updates, such as DQN whose so-called target network
    is synchronized periodically.

    Parameters
    ----------
    env : gym environment

        The main gym environment to be wrapped.


    T_start : non-negative int, optional

        The number of time steps that have passed before the start of this
        particular counter wrapper.

    episode_start : non-negative int, optional

        The number of episodes that have passed before the start of this
        particular counter wrapper.


    """
    def __init__(
            self,
            env: gym.Env,
            T_start: int = 0,
            episode_start: int = 0):

        super().__init__(env)
        self._T = T_start
        self._episode = episode_start
        self._periodic_counters = {}
        self._periodic_step_checks = {}
        self._periodic_episode_checks = {}

    @property
    def T(self):
        return self._T

    @property
    def episode(self):
        return self._episode

    @property
    def periodic_counters(self):
        return deepcopy(self._periodic_counters)

    @property
    def periodic_checks(self):
        checks = {}
        checks.update(self._periodic_step_checks)
        checks.update(self._periodic_episode_checks)
        return checks

    def add_periodic_counter(
            self,
            name: str,
            period: int,
            counter_type: str = 'step'):
        """
        Adds a periodic time-step counter with specified name.

        Parameters
        ----------
        name : str

            Name (identifier) of the periodic counter.

        period : positive int

            The period of the periodic counter.

        counter_type : 'step' or 'episode'

            Whether to count time steps or episodes. A ``'step'`` counter is
            incremented and checked when the :func:`step` method is called,
            whereas an ``'episode'`` counter is incremented and checked when
            the :func:`reset` is called.

        """
        self._periodic_counters[name] = PeriodicCounter(
            name=name, period=period, counter_type=counter_type,
            start_value=self.T)
        if counter_type == 'step':
            self._periodic_step_checks[name] = False
        elif counter_type == 'episode':
            self._periodic_episode_checks[name] = False
        else:
            raise ValueError("bad counter_type")

    def step(self, a):
        if not self._episode:
            RuntimeError(
                "Please make sure to call 'reset' before your first 'step'")
        self._T += 1
        s_next, r, done, info = self.env.step(a)
        if not isinstance(info, dict):
            info = {}
        info['T'] = self.T
        info['episode'] = self.episode
        self._periodic_step_checks = {
            k: v.increment_and_check()
            for k, v in self._periodic_counters.items()
            if v.counter_type == 'step'}
        info['periodic_checks'] = self.periodic_checks
        return s_next, r, done, info

    def reset(self):
        self._episode += 1
        self._periodic_episode_checks = {
            k: v.increment_and_check()
            for k, v in self._periodic_counters.items()
            if v.counter_type == 'episode'}
        return self.env.reset()

    def reset_counters(self):
        self._periodic_step_checks = {
            k: False for k in self._periodic_step_checks}
        self._periodic_episode_checks = {
            k: False for k in self._periodic_episode_checks}
        self._T = 0
        self._episode = 0
        for c in self._periodic_counters.values():
            c.counter = 0
