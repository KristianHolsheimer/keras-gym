import gym
import numpy as np
from PIL import Image

from ..utils import check_numpy_array


__all__ = (
    'AtariPreprocessor',
)


class AtariPreprocessor(gym.Wrapper):
    """
    Dedicated preprocessor for Atari environments.

    This preprocessing is adapted from this blog post:

        https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

    Parameters
    ----------
    num_frames : positive int, optional

        Number of frames to stack in order to build a state feature vector.

    """
    IMG_SHAPE_ORIG = (210, 160, 3)
    IMG_SHAPE = (105, 80)

    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = int(num_frames)
        self.observation_space = gym.spaces.Box(
            shape=(self.IMG_SHAPE + (self.num_frames,)),
            low=0, high=255, dtype='uint8')

    def reset(self):
        self._i = 0
        shape = (self.num_frames,) + self.IMG_SHAPE
        self._frames = np.zeros(shape, self.observation_space.dtype)
        self._s_orig = self.env.reset()
        s = self.preprocess_frame(self._s_orig)
        self._frames[...] = np.expand_dims(s, axis=0)
        s = np.transpose(self._frames, (1, 2, 0))
        return s

    def step(self, a):
        self._s_next_orig, r, done, info = self.env.step(a)
        self._add_orig_to_info_dict(info)
        s_next = self.preprocess_frame(self._s_next_orig)
        self._frames[self._i] = s_next
        self._i += 1
        self._i %= self.num_frames
        s_next = np.roll(self._frames, -self._i)  # latest frame is last
        s_next = np.transpose(s_next, (1, 2, 0))
        return s_next, r, done, info

    @classmethod
    def preprocess_frame(cls, s):
        h, w = cls.IMG_SHAPE
        check_numpy_array(s, shape=cls.IMG_SHAPE_ORIG)
        img = Image.fromarray(s)
        img = img.convert('L')  # grayscale
        img = img.resize((w, h))
        return np.array(img)

    def _add_orig_to_info_dict(self, info):
        if not isinstance(info, dict):
            assert info is None, "unexpected type for 'info' dict"
            info = {}

        if 's_orig' in info:
            info['s_orig'].append(self._s_orig)
        else:
            info['s_orig'] = [self._s_orig]

        if 's_next_orig' in info:
            info['s_next_orig'].append(self._s_next_orig)
        else:
            info['s_next_orig'] = [self._s_next_orig]
