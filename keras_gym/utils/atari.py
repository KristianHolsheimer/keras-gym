import numpy as np
from PIL import Image
from . import ArrayDeque


class AtariPreprocessor:
    """
    Dedicated preprocessor for Atari environments.

    This preprocessing is adapted from this blog post:

        https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

    Parameters
    ----------
    num_frames : positive int, optional

        Number of frames to stack in order to build a state feature vector.

    Attributes
    ----------
    deque : keras_gym.utils.ArrayDeque

        The array deque with settings:
        ``ArrayDeque(maxlen=num_frames, overflow='cycle')``

    shape : tuple of int

        Shape tuple: ``(num_frames, 105, 80)``.

    shape_flat : tuple

        Shape tuple ``(num_frames * 105 * 80,)``.

    """
    IMG_SHAPE_ORIG = (210, 160, 3)
    IMG_SHAPE = (105, 80)

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self._init_deque()

    def __call__(self, s):
        """
        Preprocess a single state observation.

        Parameters
        ----------
        s : state observation

            A single state observation of shape [210, 160, 3] and dtype
            'uint8'.

        Returns
        -------
        x : 1d array, shape: [num_frames * 105 * 80], dtype: 'uint8'

            Preprocessed state consisting of ``num_frames`` stacked frames.

        """
        h, w = self.IMG_SHAPE
        assert s.shape == self.IMG_SHAPE_ORIG, "bad shape"
        img = Image.fromarray(s)
        img = img.convert('L')  # grayscale
        img = img.resize((w, h))
        self.deque.append(np.array(img))
        x = self.deque.array.transpose((1, 2, 0))  # height x width x channels
        assert x.shape == self.shape, "bad shape"
        return x

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self.IMG_SHAPE + (self.num_frames,)
        return self._shape

    @property
    def shape_flat(self):
        if not hasattr(self, '_shape_flat'):
            self._shape_flat = (np.prod(self.shape),)
        return self._shape_flat

    def _init_deque(self):
        self.deque = ArrayDeque(
            shape=self.IMG_SHAPE, maxlen=self.num_frames, overflow='cycle',
            dtype='uint8')

        # initialize with zero-padding
        for _ in range(self.num_frames):
            self.deque.append(np.zeros(self.IMG_SHAPE, dtype='uint8'))
