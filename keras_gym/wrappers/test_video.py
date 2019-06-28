import gym
import numpy as np

from .video import ImagePreprocessor, FrameStacker


class MockEnv:
    def __init__(self, shape, num_actions):
        self.shape = shape
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(
            shape=self.shape, low=0, high=255, dtype='uint8')
        self.reward_range = (-np.infty, np.infty)
        self.metadata = None
        self.spec = None

    def reset(self):
        self.i = 0
        s = np.full(self.shape, 100 + self.i)
        return s

    def step(self, a):
        self.i += 1
        s_next = np.full(self.shape, 100 + self.i)
        r = -(100 + self.i)
        done = self.i >= 10
        info = {'i': self.i}
        return s_next, r, done, info


class TestImagePreprocessor:

    def test_shape(self):
        env0 = gym.make('PongDeterministic-v4')
        env1 = ImagePreprocessor(env0, height=105, width=80, grayscale=False)
        env2 = ImagePreprocessor(env0, height=105, width=80, grayscale=True)

        s = env0.reset()
        assert env0.observation_space.shape == (210, 160, 3)
        assert s.shape == (210, 160, 3)

        s = env1.reset()
        assert env1.observation_space.shape == (105, 80, 3)
        assert s.shape == (105, 80, 3)

        s = env2.reset()
        assert env2.observation_space.shape == (105, 80)
        assert s.shape == (105, 80)


class TestFrameStacker:
    def test_shape(self):
        env0 = gym.make('PongDeterministic-v4')
        env1 = ImagePreprocessor(env0, height=105, width=80, grayscale=True)
        env2 = FrameStacker(env0, num_frames=5)
        env3 = FrameStacker(env1, num_frames=5)

        s = env2.reset()
        assert env2.observation_space.shape == (210, 160, 3, 5)
        assert s.shape == (210, 160, 3, 5)

        s = env3.reset()
        assert env3.observation_space.shape == (105, 80, 5)
        assert s.shape == (105, 80, 5)

    def test_order(self):
        env0 = MockEnv(shape=(2, 3), num_actions=2)
        env1 = FrameStacker(env0, num_frames=5)

        s = env1.reset()
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 100, 100, 100])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 100, 100, 101])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 100, 101, 102])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 101, 102, 103])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 101, 102, 103, 104])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [101, 102, 103, 104, 105])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [102, 103, 104, 105, 106])

        s = env1.reset()
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 100, 100, 100])

        s, r, done, info = env1.step(0)
        assert s.shape == (2, 3, 5)
        np.testing.assert_array_equal(s[0, 0, :], [100, 100, 100, 100, 101])
