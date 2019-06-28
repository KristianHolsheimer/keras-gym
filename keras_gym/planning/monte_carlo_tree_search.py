from copy import deepcopy

import numpy as np

from ..base.mixins import NumActionsMixin, RandomStateMixin
from ..base.errors import LeafNodeError, NotLeafNodeError, EpisodeDoneError
from ..utils import argmax

__all__ = (
    'MCTSNode',
)


class MCTSNode(NumActionsMixin, RandomStateMixin):
    def __init__(
            self,
            state_id,
            actor_critic,
            tau=1.0,
            v_resign=-1.0,
            c_puct=1.414,
            random_seed=None):

        self.state_id = state_id
        self.actor_critic = actor_critic
        self.tau = tau
        self.v_resign = v_resign
        self.c_puct = c_puct
        self.random_seed = random_seed  # also sets self.random

        # these are here for convenience
        self.env = deepcopy(self.actor_critic.env)
        self.env.set_state(self.state_id)
        self.state = self.env.state
        self.done = self.env.done

        # these are set/updated dynamically
        self.parent_node = None
        self.parent_action = None
        self.children = {}
        self.is_leaf = True
        self.v_max = -np.inf
        self.v = None
        self.P = None

    def __repr__(self):
        s = "MCTSNode('{}', v={:s} done={}".format(
            self.state_id, self._str(self.v, length=5, suffix=','),
            self._str(self.done, suffix=')', length=5))
        return s

    @property
    def is_root(self):
        return self.parent_node is None

    @property
    def U(self):
        if self.is_leaf:
            U = None
        else:
            # PUCT: U(s,a) = P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))
            U = self.c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
            U[self.D] = 0
        return U

    @property
    def Q(self):
        if self.is_leaf:
            Q = None
        else:
            Q = self.W / (self.N + 1e-16)
            Q[self.D] = self.env.win_reward
        return Q

    def search(self, n=512):
        for _ in range(n):
            leaf_node = self.select()
            v = leaf_node.v if leaf_node.done else leaf_node.expand()
            leaf_node.backup(v)

    def play(self, tau=None):
        if self.is_leaf:
            raise LeafNodeError(
                "cannot play from a leaf node; must search first")

        if tau is None:
            tau = self.tau

        # construct pi(a|s) = N(s,a)^1/tau / N(s)
        if tau < 0.1:
            # no need to compute pi if tau is very small
            a = argmax(self.N, random_state=self.random)
        else:
            pi = np.power(self.N, 1 / tau)
            pi /= np.sum(pi)
            a = self.random.choice(self.num_actions, p=pi)

        # this will become the new root node
        child = self.children[a]

        # update env
        s_next, r, done, info = self.env.step(a)
        assert child.state_id == info['state_id']

        # switch to new root node
        self.__dict__.update(child.__dict__)
        self.parent_node = None
        self.parent_action = None

        return self.state, a, r, done or self.v_max < self.v_resign

    def select(self):
        if self.is_leaf:
            return self

        # pick action according to PUCT algorithm
        a = argmax(
            (self.Q + self.U)[self.env.available_actions],
            random_state=self.random)
        child = self.children[a]

        # recursively traverse down the tree
        return child.select()

    def expand(self):
        """
        Expand tree, i.e. promote leaf node to a non-leaf node.

        """
        if not self.is_leaf:
            raise NotLeafNodeError(
                "node is not a leaf node; cannot expand node more than once")
        if self.done:
            raise EpisodeDoneError("cannot expand further; episode is done")

        self.P, v = self.actor_critic.proba(self.state)
        if self.v is None:
            self.v = float(v)

        for a in self.env.available_actions:
            s_next, r, done, info = self.env.step(a)
            child = MCTSNode(info['state_id'], self.actor_critic)
            child.parent_node = self
            child.parent_action = a
            if done:
                self.D[a] = True
                child.v = -r  # note: flip sign for 'opponent'
            self.children[a] = child
            self.env.set_state(self.state_id)  # reset state to root

        self.is_leaf = False

        return self.v

    @property
    def N(self):
        if not hasattr(self, '_N'):
            self._N = np.zeros(self.num_actions, dtype='int')
        return self._N

    @property
    def W(self):
        if not hasattr(self, '_W'):
            self._W = np.zeros(self.num_actions, dtype='float')
        return self._W

    @property
    def D(self):
        if not hasattr(self, '_D'):
            self._D = np.zeros(self.num_actions, dtype='bool')
        return self._D

    def backup(self, v, a=None):
        if self.is_leaf and not self.done:
            raise LeafNodeError(
                "node is a leaf node; cannot backup before expanding")

        self.v_max = max(self.v_max, v)
        if a is not None:
            self.N[a] += 1
            self.W[a] += v

        # recursively traverse up the tree
        if not self.is_root:
            # notice that we flip sign for 'opponent'
            self.parent_node.backup(-v, self.parent_action)

    def show(self, depth=np.inf, prefix='', suffix=''):
        if depth == 0:
            return
        print(prefix + str(self) + suffix)
        if self.children and depth > 1:
            print()
        for a, child in self.children.items():
            child.show(
                depth=(depth - 1),
                prefix=(prefix + "    "),
                suffix=(
                    "  a={:d}  Q={:s}  U={:s}  N={:s}"
                    .format(
                        a, self._str(self.Q[a]), self._str(self.U[a]),
                        self._str(self.N[a]))))
            if a == 6 and depth > 1:
                print()

    @staticmethod
    def _str(x, suffix='', length=5):
        if isinstance(x, (float, np.float32, np.float64)):
            x = '{:g}'.format(x)
        s = str(x)[:length].strip() + suffix
        s += ' ' * max(0, length + len(suffix) - len(s))
        return s
