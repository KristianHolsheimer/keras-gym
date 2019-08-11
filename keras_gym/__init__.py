# flake8: noqa

# ensure that all submodules are visible
from . import (
    base, caching, envs, function_approximators, losses, planning, policies,
    wrappers, utils)

# Expose some commonly used classes to the package root:
from .function_approximators import predefined
from .function_approximators.generic import (
    FunctionApproximator, V, QTypeI, QTypeII, SoftmaxPolicy,
    ConjointActorCritic)
from .function_approximators.actor_critic import ActorCritic
from .policies import EpsilonGreedy, RandomPolicy, UserInputPolicy
from .utils import render_episode, enable_logging
