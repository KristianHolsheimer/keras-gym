import pytest

import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from ..wrappers.env_wrappers import CounterWrapper, TransitionWrapper


class TestTransitionWrapper:
    def test_step_before_reset(self):
        msg = r"must reset\(\) env before running step\(\)"
        env = FrozenLakeEnv(is_slippery=False)
        env = TransitionWrapper(env, state_preprocessor=(lambda s: [s]))
        with pytest.raises(RuntimeError, match=msg):
            env.step(0)

    def test_transition_length_5(self):
        n = 5
        env = TransitionWrapper(
            FrozenLakeEnv(is_slippery=False),
            state_preprocessor=(lambda s: [s]),
            bootstrap_n=n)
        env.seed(13)
        env.reset()
        A = [2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 1, 2]  # episode has 14 steps
        for t, a in enumerate(A, 1):
            s_next, r, done, info = env.step(a)
            if done:
                assert len(env.transitions) == t
                break
            elif t < n:
                assert len(env.transitions) == 0
            else:
                assert len(env.transitions) == t - n

    def test_transition_length_1(self):
        n = 1
        env = TransitionWrapper(
            FrozenLakeEnv(is_slippery=False),
            state_preprocessor=(lambda s: [s]),
            bootstrap_n=n)
        env.seed(13)
        env.reset()
        A = [2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 1, 2]  # episode has 14 steps
        for t, a in enumerate(A, 1):
            s_next, r, done, info = env.step(a)
            if done:
                assert len(env.transitions) == t
                break
            elif t < n:
                assert len(env.transitions) == 0
            else:
                assert len(env.transitions) == t - n

    def test_transition_length_20(self):
        n = 20  # effectively Monte Carlo
        env = TransitionWrapper(
            FrozenLakeEnv(is_slippery=False),
            state_preprocessor=(lambda s: [s]),
            bootstrap_n=n)
        env.seed(13)
        env.reset()
        A = [2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 1, 2]  # episode has 14 steps
        for t, a in enumerate(A, 1):
            s_next, r, done, info = env.step(a)
            if done:
                assert len(env.transitions) == t
                break
            elif t < n:
                assert len(env.transitions) == 0
            else:
                assert len(env.transitions) == t - n

    def test_transition_online_popleft(self):
        n = 5
        env = TransitionWrapper(
            FrozenLakeEnv(is_slippery=False),
            state_preprocessor=(lambda s: [s]),
            bootstrap_n=n)
        env.seed(13)
        env.reset()
        A = [2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 1, 0, 2, 2]  # 16 steps
        T = list(range(len(A)))  # ordered
        i = 0
        for t, a in enumerate(A, 1):
            s_next, r, done, info = env.step(a)

            while env.transitions:
                transition = env.transitions.popleft()
                assert transition.A == A[T[i]]
                i += 1

            if done:
                break

    def test_transition_online_pop(self):
        n = 5
        env = TransitionWrapper(
            FrozenLakeEnv(is_slippery=False),
            state_preprocessor=(lambda s: [s]),
            bootstrap_n=n)
        env.seed(13)
        env.reset()
        A = [2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 1, 0, 2, 2]  # 16 steps
        T = list(range(len(A)))
        T[-(n + 1):] = reversed(T[-(n + 1):])  # last n+1 reversed
        i = 0
        for t, a in enumerate(A, 1):
            s_next, r, done, info = env.step(a)

            while env.transitions:
                transition = env.transitions.pop()
                assert transition.A[0] == A[T[i]]
                i += 1

            if done:
                break


class TestCounterWrapper:
    env = gym.make('CartPole-v0')
    env = CounterWrapper(env)
    env.add_periodic_counter('T13', period=13)
    env.add_periodic_counter('e03', period=3, counter_type='episode')

    def test_base_counters(self):
        self.env.reset_counters()

        T = 0
        for episode in range(1, 6):
            self.env.reset()
            assert self.env.episode == episode

            for t in range(self.env.spec.max_episode_steps):
                a = self.env.action_space.sample()
                s_next, r, done, info = self.env.step(a)

                T += 1
                assert self.env.T == T

                if done:
                    break

    def test_periodic_counters(self):
        self.env.reset_counters()

        for episode in range(1, 10):
            self.env.reset()
            assert self.env.periodic_checks['e03'] == (episode % 3 == 0)

            for t in range(self.env.spec.max_episode_steps):
                a = self.env.action_space.sample()
                s_next, r, done, info = self.env.step(a)

                check = self.env.T % 13 == 0
                assert self.env.periodic_checks['T13'] == check

                if done:
                    break
