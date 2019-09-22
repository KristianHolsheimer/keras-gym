import numpy as np

from ..base.errors import UnavailableActionError, ActionSpaceError
from ..base.mixins import RandomStateMixin, ActionSpaceMixin
from ..policies.base import BasePolicy

__all__ = (
    'RandomPolicy',
    'UserInputPolicy',
)


class RandomPolicy(BasePolicy, RandomStateMixin, ActionSpaceMixin):
    """
    Value-based policy to select actions using epsilon-greedy strategy.

    Parameters
    ----------
    env : gym environment

       The gym environment is used to sample from the action space.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, env, random_seed=None):
        self.env = env
        self.random_seed = random_seed  # sets self.random in RandomStateMixin

    def __call__(self, s):
        return self.env.action_space.sample()

    def greedy(self, s):
        return self(s)

    def dist_params(self, s):
        if self.action_space_is_discrete:
            return np.ones(self.num_actions) / self.num_actions

        if self.action_space_is_box:
            mu = np.zeros(self.actions_ndim)          # zero mean
            logvar = 10 * np.ones(self.actions_ndim)  # large variance
            return mu, logvar

        raise ActionSpaceError(
            "method RandomPolicy.dist_params() is not yet implemented for "
            "action spaces of type: {}"
            .format(self.env.action_space.__class__.__name__))


class UserInputPolicy(BasePolicy, ActionSpaceMixin):
    """
    A policy that prompts the user to take an action.

    Parameters
    ----------
    env : gym environment

       The gym environment is used to sample from the action space.

    render_before_prompt : bool, optional

        Whether to render the env before prompting the user to pick an action.

    """
    def __init__(self, env, render_before_prompt=False):
        self.env = env
        self.render_before_prompt = bool(render_before_prompt)

    def __call__(self, s):
        actions = ", ".join(map(str, range(self.num_actions)))
        if self.render_before_prompt:
            self.env.render()

        for attempt in range(1, 4):  # 3 attempts
            try:
                a = input("Pick action from {{{}}}: ".format(actions))
                return int(a)
            except ValueError:
                print(
                    "ValueError: invalid action, try again (attempt {:d} of 3)"
                    "...".format(attempt))

        raise UnavailableActionError("a = {}".format(a))

    def greedy(self, s):
        return self(s)

    def dist_params(self, s):
        raise NotImplementedError('UserInputPolicy.dist_params')
    dist_params.__doc__ = ""
