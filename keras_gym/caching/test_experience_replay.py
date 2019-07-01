import gym
import numpy as np
from .experience_replay import ExperienceReplayBuffer


class MockEnv:
    action_space = gym.spaces.Discrete(7)

    def __init__(self, num_frames):
        self.num_frames = num_frames


class TestExperienceReplayBuffer:
    N = 7
    S = np.expand_dims(np.arange(N), axis=1)
    A = S[:, 0] % 100
    R = S[:, 0]
    D = np.zeros(N, dtype='bool')
    D[-1] = True
    EPISODE = list(zip(S, A, R, D))

    def test_add(self):
        buffer = ExperienceReplayBuffer(MockEnv(1), capacity=17)
        for i, (s, a, r, done) in enumerate(self.EPISODE, 1):
            buffer.add(s + 100, a, r + 100, done, episode_id=1)
            assert len(buffer) == max(0, i - buffer.bootstrap_n)

        np.testing.assert_array_equal(
            buffer._e[:7],
            [1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            buffer._d[:7].astype('int'),
            [0, 0, 0, 0, 0, 0, 1])

        for i, (s, a, r, done) in enumerate(self.EPISODE, i + 1):
            buffer.add(s + 200, a, r + 200, done, episode_id=2)
            assert len(buffer) == max(0, i - buffer.bootstrap_n)

        np.testing.assert_array_equal(
            buffer._e[:14],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(
            buffer._d[:14].astype('int'),
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

        for i, (s, a, r, done) in enumerate(self.EPISODE, i + 1):
            buffer.add(s + 300, a, r + 300, done, episode_id=3)
            assert len(buffer) == np.clip(i - buffer.bootstrap_n, 0, 17)

        # buffer wraps around and overwrites oldest transitions
        np.testing.assert_array_equal(
            buffer._e,
            [3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])
        np.testing.assert_array_equal(
            buffer._d.astype('int'),
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            buffer._p,
            [
                [0, 0, 0, 0, 1, 0, 0],  # a=4
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [0, 0, 0, 1, 0, 0, 0],  # a=3
                [0, 0, 0, 0, 1, 0, 0],  # a=4
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [1, 0, 0, 0, 0, 0, 0],  # a=0
                [0, 1, 0, 0, 0, 0, 0],  # a=1
                [0, 0, 1, 0, 0, 0, 0],  # a=2
                [0, 0, 0, 1, 0, 0, 0],  # a=3
                [0, 0, 0, 0, 1, 0, 0],  # a=4
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [1, 0, 0, 0, 0, 0, 0],  # a=0
                [0, 1, 0, 0, 0, 0, 0],  # a=1
                [0, 0, 1, 0, 0, 0, 0],  # a=2
                [0, 0, 0, 1, 0, 0, 0],  # a=3
            ])

    def test_sample(self):
        buffer = ExperienceReplayBuffer(
            env=MockEnv(num_frames=3), capacity=17, random_seed=13,
            batch_size=16, bootstrap_n=2)

        for ep in (1, 2, 3):
            for s, a, r, done in self.EPISODE:
                buffer.add(
                    s[[0, 0, 0]] + ep * 100, a, r + ep * 100, done,
                    episode_id=ep)

        # quickly check content, just to be safe
        np.testing.assert_array_equal(
            buffer._p,
            [
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [0, 0, 1, 0, 0, 0, 0],  # a=2
                [0, 0, 0, 1, 0, 0, 0],  # a=3
                [0, 0, 0, 0, 1, 0, 0],  # a=4
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [1, 0, 0, 0, 0, 0, 0],  # a=0
                [0, 1, 0, 0, 0, 0, 0],  # a=1
                [0, 0, 1, 0, 0, 0, 0],  # a=2
                [0, 0, 0, 1, 0, 0, 0],  # a=3
                [0, 0, 0, 0, 1, 0, 0],  # a=4
                [0, 0, 0, 0, 0, 1, 0],  # a=5
                [0, 0, 0, 0, 0, 0, 1],  # a=6
                [1, 0, 0, 0, 0, 0, 0],  # a=0
                [0, 1, 0, 0, 0, 0, 0],  # a=1
                [0, 0, 1, 0, 0, 0, 0],  # a=2
                [0, 0, 0, 1, 0, 0, 0],  # a=3
                [0, 0, 0, 0, 1, 0, 0],  # a=4
            ])

        transitions = [[[102, 103, 104], 0.9801, [104, 105, 106]],  # bootstrap
                       [[203, 204, 205], 0.0000, [205, 206, 300]],
                       [[200, 200, 201], 0.9801, [201, 202, 203]],  # bootstrap
                       [[102, 103, 104], 0.9801, [104, 105, 106]],  # bootstrap
                       [[104, 105, 106], 0.0000, [106, 200, 201]],
                       [[202, 203, 204], 0.9801, [204, 205, 206]],  # bootstrap
                       [[300, 300, 300], 0.9801, [300, 301, 302]],  # bootstrap
                       [[203, 204, 205], 0.0000, [205, 206, 300]],
                       [[103, 104, 105], 0.0000, [105, 106, 200]],
                       [[203, 204, 205], 0.0000, [205, 206, 300]],
                       [[104, 105, 106], 0.0000, [106, 200, 201]],
                       [[102, 103, 104], 0.9801, [104, 105, 106]],  # bootstrap
                       [[200, 200, 201], 0.9801, [201, 202, 203]],  # bootstrap
                       [[200, 200, 200], 0.9801, [200, 201, 202]],  # bootstrap
                       [[300, 300, 301], 0.9801, [301, 302, 303]],  # bootstrap
                       [[203, 204, 205], 0.0000, [205, 206, 300]]]

        S, P, Rn, In, S_next, P_next = buffer.sample()
        np.testing.assert_array_equal(In, [tr[1] for tr in transitions])
        np.testing.assert_array_equal(
            S[:, 0, :], [tr[0] for tr in transitions])
        np.testing.assert_array_equal(
            S_next[:, 0, :], [tr[2] for tr in transitions])

        # check if actions are separate by bootstrap_n steps
        A = np.argmax(P, axis=1)
        A_next = np.argmax(P_next, axis=1)
        for a, i_next, a_next in zip(A, In, A_next):
            if i_next != 0:
                assert a_next - a == buffer.bootstrap_n

        # check if states and actions are aligned
        np.testing.assert_array_equal(S[:, 0, -1] % 100, A)
        np.testing.assert_array_equal(S_next[:, 0, -1] % 100, A_next)

    def test_shape(self):
        buffer = ExperienceReplayBuffer(
            env=MockEnv(num_frames=3), capacity=17, batch_size=5,
            random_seed=13)

        for ep in (1, 2, 3):
            for i, (_, a, r, done) in enumerate(self.EPISODE):
                s = 100 * ep + i * np.ones((11, 13, 3), dtype='int')
                buffer.add(s, a, r, done, ep)

        S, A, Rn, In, S_next, A_next = buffer.sample()
        assert S.shape == (5, 11, 13, 3)

        # check if all frames come from the same episode
        np.testing.assert_array_equal(
            S[:, 0, 0, :],     # look at upper-left pixel only
            [[304, 305, 306],
             [203, 204, 205],
             [304, 305, 306],
             [200, 200, 201],  # note: first frame is repeated
             [104, 105, 106]])
