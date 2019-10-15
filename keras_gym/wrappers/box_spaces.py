import gym
import numpy as np

from ..base.errors import ActionSpaceError
from ..base.mixins import ActionSpaceMixin, AddOrigToInfoDictMixin
from ..utils import reals_to_box_np


__all__ = (
    'BoxActionsToReals',
)


class BoxActionsToReals(gym.Wrapper, ActionSpaceMixin, AddOrigToInfoDictMixin):
    """

    This wrapper decompactifies a :class:`Box <gym.spaces.Box>` action space to
    the reals. This is required in order to be able to use a
    :class:`GaussianPolicy <keras_gym.GaussianPolicy>`.

    In practice, the wrapped environment expects the input action
    :math:`a_\\text{real}\\in\\mathbb{R}^n` and then it compactifies it back to
    a Box of the right size:

    .. math::

        a_\\text{box}\\ =\\
            \\text{low} + (\\text{high}-\\text{low})
                \\times\\text{sigmoid}(a_\\text{real})

    Technically, the transformed space is still a Box, but that's only because
    we assume that the values lie between large but finite bounds,
    :math:`a_\\text{real}\\in[10^{-15}, 10^{15}]^n`.

    """
    def __init__(self, env):
        super().__init__(env)
        shape = self.env.action_space.shape
        dtype = self.env.action_space.dtype
        self.action_space = gym.spaces.Box(
            low=np.full(shape, -1e15, dtype),
            high=np.full(shape, 1e15, dtype))

        if not self.action_space_is_box:
            raise ActionSpaceError(
                "BoxActionsToReals is only implemented for Box action spaces")

    def step(self, a):
        assert self.action_space.contains(a)
        self._a_orig = reals_to_box_np(a, self.env.action_space)
        s_next, r, done, info = super().step(self._a_orig)
        self._add_a_orig_to_info_dict(info)
        return s_next, r, done, info
