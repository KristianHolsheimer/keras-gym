import gym

from .video import ImagePreprocessor, FrameStacker


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
