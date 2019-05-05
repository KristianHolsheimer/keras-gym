import gym

from ..wrappers.env_wrappers import CounterWrapper


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
