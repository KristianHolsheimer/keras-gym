import gym
from copy import deepcopy


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
