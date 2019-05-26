
from ..utils import is_vfunction, is_qfunction, is_policy
from ..base.policy import BasePolicy
from ..base.mixins import NumActionsMixin
from ..base.function_approximators.generic import BaseFunctionApproximator


class ActorCritic(BaseFunctionApproximator, BasePolicy, NumActionsMixin):
    def __init__(self, policy, value_function):
        self.policy = policy
        self.value_function = value_function

        # inherit some attrs
        self.env = self.value_function.env
        self._cache = self.value_function._cache

        self._check_function_types()

    def update(self, s, a, r, done):
        """
        Update both actor and critic.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action that was taken.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        """
        assert self.env.observation_space.contains(s)
        assert self.env.action_space.contains(a)
        self._cache.add(s, a, r, done)

        # eager updates
        while self._cache:
            self.batch_update(*self._cache.pop())  # pop with batch_size=1

    def batch_update(self, S, A, Rn, I_next, S_next, A_next=None):
        """
        Update both actor and critic on a batch of transitions.

        Parameters
        ----------
        S : nd array, shape: [batch_size, ...]

            A batch of state observations.

        A : 1d array, dtype: int, shape: [batch_size]

            A batch of actions that were taken.

        Rn : 1d array, dtype: float, shape: [batch_size]

            A batch of partial returns. For example, in n-step bootstrapping
            this is given by:

            .. math::

                R^{(n)}_t\\ =\\ R_t + \\gamma\\,R_{t+1} + \\dots
                    \\gamma^{n-1}\\,R_{t+n-1}

            In other words, it's the non-bootstrapped part of the n-step
            return.

        I_next : 1d array, dtype: float, shape: [batch_size]

            A batch bootstrapping factor. For instance, in n-step bootstrapping
            this is given by :math:`I_t=\\gamma^n` if the episode is ongoing
            and :math:`I_t=0` otherwise. This allows us to write the
            bootstrapped target as :math:`G^{(n)}_t=R^{(n)}_t+I_tQ(S_{t+n},
            A_{t+n})`.

        S_next : nd array, shape: [batch_size, ...]

            A batch of next-state observations.

        A_next : 1d array, dtype: int, shape: [batch_size], optional

            A batch of next-actions that were taken. This is only required for
            SARSA (on-policy) updates.

        """
        # TODO: This will be optimized such that S is only fed into the graph
        # once instead of three times.
        V = self.value_function.batch_eval(S, use_target_model=False)
        V_next = self.value_function.batch_eval(S_next, use_target_model=True)
        G = Rn + I_next * V_next

        self.policy.batch_update(S, A, G - V)
        self.value_function.batch_update(S, Rn, I_next, S_next)

    def __call__(self, s):
        return self.policy(s)

    def batch_eval(self, S):
        return self.policy.batch_eval(S)

    def greedy(self, s):
        return self.policy.greedy(s)

    def proba(self, s):
        return self.policy.proba(s)

    def _check_function_types(self):
        if not is_vfunction(self.value_function):
            if is_qfunction(self.value_function):
                raise NotImplementedError(
                    "ActorCritic hasn't been yet implemented for Q-functions, "
                    "please let me know is you need this; for the time being, "
                    "please use V-function instead.")
        if not is_policy(self.policy, check_updateable=True):
            raise TypeError("expected an updateable policy")
        if self.policy.env != self.value_function.env:
            raise ValueError(
                "the envs of policy and value_function do not match")
