from ..base.errors import UnavailableActionError
from ..base.mixins import RandomStateMixin, NumActionsMixin
from ..policies.base import BasePolicy

__all__ = (
    'RandomPolicy',
    'UserInputPolicy',
)


class RandomPolicy(BasePolicy, RandomStateMixin, NumActionsMixin):
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

    def proba(self, s):
        return 1.0 / self.num_actions


class UserInputPolicy(BasePolicy, NumActionsMixin):
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
                a = input(
                    "Pick action from {{{}}}: ".format(actions))
                return int(a)
            except ValueError:
                print(
                    "ValueError: invalid action, try again (attempt {:d} of 3)"
                    "...".format(attempt))

        raise UnavailableActionError("a = {}".format(a))

    def greedy(self, s):
        return self(s)

    def proba(self, s):
        raise NotImplementedError('UserInputPolicy.proba')
    proba.__doc__ = ""
