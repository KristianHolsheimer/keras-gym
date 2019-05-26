import gym
import numpy as np
from PIL import Image

from ..utils import check_numpy_array
from ..base.mixins import AddOrigStateToInfoDictMixin
from ..base.errors import NumpyArrayCheckError


__all__ = (
    'ImagePreprocessor',
    'FrameStacker',
)


class ImagePreprocessor(gym.Wrapper, AddOrigStateToInfoDictMixin):
    """
    Preprocessor for images.

    This preprocessing is adapted from this blog post:

        https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26


    Parameters
    ----------
    env : gym environment

        A gym environment.

    height : positive int

        Output height (number of pixels).

    width : positive int

        Output width (number of pixels).

    grayscale : bool, optional

        Whether to convert RGB image to grayscale.

    assert_input_shape : shape tuple, optional

        If provided, the preprocessor will assert the given input shape.

    """
    def __init__(
            self, env, height, width,
            grayscale=True,
            assert_input_shape=None):

        super().__init__(env)

        self.height = int(height)
        self.width = int(width)
        self.grayscale = bool(grayscale)

        # check input shape?
        self.assert_input_shape = assert_input_shape
        if self.assert_input_shape is not None:
            self.assert_input_shape = tuple(self.assert_input_shape)

        # check original shape / dtype
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype
        assert len(shape) == 3, "bad shape: {}".format(shape)
        assert shape[2] == 3, "bad shape: {}".format(shape)
        assert dtype == 'uint8', "bad dtype: {}".format(dtype)

        # update observation space
        if self.grayscale:
            shape = (self.height, self.width)
        else:
            shape = (self.height, self.width, shape[2])
        self.observation_space = gym.spaces.Box(
            shape=shape, low=0, high=255, dtype='uint8')

    def _preprocess_frame(self, s):
        check_numpy_array(s, shape=self.assert_input_shape)
        img = Image.fromarray(s)
        if self.grayscale:
            img = img.convert('L')
        img = img.resize((self.width, self.height))
        return np.array(img)

    def reset(self):
        self._s_orig = self.env.reset()
        s = self._preprocess_frame(self._s_orig)   # shape: [h, w]
        return s

    def step(self, a):
        self._s_next_orig, r, done, info = self.env.step(a)
        self._add_orig_to_info_dict(info)
        s_next = self._preprocess_frame(self._s_next_orig)
        return s_next, r, done, info


class FrameStacker(gym.Wrapper, AddOrigStateToInfoDictMixin):
    """
    Stack multiple frames into one state observation.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    num_frames : positive int, optional

        Number of frames to stack in order to build a state feature vector.

    """
    def __init__(
            self, env,
            num_frames=4):

        super().__init__(env)

        self.num_frames = int(num_frames)

        s = self.env.observation_space.sample()
        check_numpy_array(s, dtype='uint8')
        if s.ndim == 2:
            self._perm = (1, 2, 0)
        elif s.ndim == 3:
            self._perm = (1, 2, 3, 0)
        else:
            NumpyArrayCheckError(
                "expected ndim equal to 2 or 3, got shape: {}".format(s.shape))

        # update observation space
        shape = s.shape + (self.num_frames,)
        self.observation_space = gym.spaces.Box(
            shape=shape, low=0, high=255, dtype='uint8')

    def reset(self):
        frame_shape = tuple(self.env.observation_space.shape)  # [h, w, c?]
        shape = (self.num_frames,) + frame_shape               # [f, h, w, c?]
        self._frames = np.zeros(shape, self.observation_space.dtype)
        self._s_orig = self.env.reset()             # shape: [h, w, c?]
        s = np.expand_dims(self._s_orig, axis=0)    # shape: [1, h, w, c?]
        self._frames[...] = s                       # broadcast along axis=0
        s = np.transpose(self._frames, self._perm)  # to shape: [h, w, c?, f]
        return s

    def step(self, a):
        self._s_next_orig, r, done, info = self.env.step(a)
        self._add_orig_to_info_dict(info)
        self._frames = np.roll(self._frames, -1, axis=0)
        self._frames[-1] = self._s_next_orig
        s_next = np.transpose(self._frames, self._perm)  # shape: [h, w, c?, f]
        return s_next, r, done, info
