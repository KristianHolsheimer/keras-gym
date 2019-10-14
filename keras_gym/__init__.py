# flake8: noqa
__version__ = '0.2.15'

from .base.patches import _monkey_patch_tensorflow
_monkey_patch_tensorflow()

# ensure that all submodules are visible
from . import (
    base, caching, envs, function_approximators, losses, planning, policies,
    wrappers, utils)

# Expose some commonly used classes to the package root:
from .function_approximators import (
    predefined, FunctionApproximator, V, QTypeI, QTypeII, SoftmaxPolicy,
    GaussianPolicy, ActorCritic)
from .policies import EpsilonGreedy, RandomPolicy, UserInputPolicy
from .utils import render_episode, enable_logging
