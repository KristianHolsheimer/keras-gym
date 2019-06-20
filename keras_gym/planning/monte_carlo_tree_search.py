from copy import deepcopy

import numpy as np

from ..base.mixins import NumActionsMixin
from ..base.errors import NotLeafNodeError


__all__ = (
    'StateNode',
    'ActionEdge',
)


class StateNode(NumActionsMixin):
    def __init__(self, state_id, actor_critic):
        self.state_id = state_id
        self.actor_critic = actor_critic
        self.env = deepcopy(self.actor_critic.env)
        self.env.set_state(self.state_id)
        self.state = self.env.state
        self.edges = []  # outgoing edges
        self.value = None

    def __repr__(self):
        return "StateNode('{}')".format(self.state_id)

    @property
    def is_leaf(self):
        return self.value is None

    def expand(self):
        """
        Expand tree, i.e. promote leaf node to a non-leaf node.

        """
        if not self.is_leaf:
            raise NotLeafNodeError(
                "node is not a leaf node; cannot expand node more than once")

        p, v = self.actor_critic(self.state)
        for a in self.env.available_actions:
            s_next, r, done, info = self.env.step(a)
            state_node_next = StateNode(info['state_id'], self.actor_critic)
            edge = ActionEdge(self, a, state_node_next, p[a], v)
            self.edges.append(edge)
            self.env.set_state(self.state_id)  # reset state

        self.value = v


class ActionEdge(NumActionsMixin):
    def __init__(self, state_node, action, state_node_next, p_a, v):
        assert isinstance(state_node, StateNode)
        assert state_node.env.action_space.contains(action)
        assert isinstance(state_node_next, StateNode)
        assert isinstance(p_a, (float, np.float32, np.float64))

        # identifiers
        self.state_node = state_node
        self.action = action
        self.state_node_next = state_node_next

        # for convenience
        self.env = self.state_node.env

        # values
        self.W = v
        self.N = 1
        self.P = p_a  # this prior is never updated

    def __repr__(self):
        return "ActionEdge('{}', a={}, N={}, Q={}, P={})".format(
            self.state_node.state_id, self.action, self.N, self.Q, self.P)

    @property
    def Q(self):
        return self.W / self.N

    def update(self, v):
        self.W += v
        self.N += 1
