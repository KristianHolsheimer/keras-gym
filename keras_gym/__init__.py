# flake8: noqa
__version__ = '0.2.16'

# ugly workarounds
from .base.patches import run
run()
del run


# ensure that all submodules are visible
from . import (
    base, caching, envs, core, losses, planning, policies,
    wrappers, utils)

# Expose some commonly used classes to the package root:
from .core import (
    predefined, FunctionApproximator, V, QTypeI, QTypeII, SoftmaxPolicy,
    GaussianPolicy, ActorCritic, SoftActorCritic)
from .policies import EpsilonGreedy, RandomPolicy, UserInputPolicy
from .utils import render_episode, enable_logging
