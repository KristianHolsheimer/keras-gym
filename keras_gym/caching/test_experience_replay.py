import pytest
import numpy as np
from .experience_replay import ExperienceReplayBuffer


class TestExperienceReplayBuffer:
    N = 17
    S = np.expand_dims(np.arange(N), axis=1)
    A = S[:, 0] % 4

    def test_expected(self):
        buffer = ExperienceReplayBuffer(capacity=7)
        print(self.S, self.A)
