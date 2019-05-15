.. automodule:: keras_gym.value_functions


Value Functions
===============

Value functions play a central role in reinforcement learning. For instance, in
value-based RL, we derive a policy from a state-action value function
:math:`Q(s,a)`. In other situations such as actor-critic type approaches we
make heavy use of value functions to estimate the advantage function
:math:`\mathcal{A}(s,a) = Q(s,a) - V(s)`.


State Value Functions
---------------------

The simplest type of value function :math:`V(s)` evaluates the value of a state
:math:`s`, which is the amount of future return expected to be collected from
that state onwards.

This type of value function cannot be used to select actions directly, but it
may be used e.g. in actor-critic scenarios to assist the policy object in their
learning process.


State-Action Value Functions
----------------------------

We can also break down the value function by actions :math:`Q(s, a)`, which
represents the amount of future return expected to be collected from the
state-action pair :math:`(s,a)`.

For a discrete action space, there are two ways to implement Q-function, which
we call :term:`type-I <type-I state-action value function>` and :term:`type-II
<type-II state-action value function>` Q-functions. A type-I Q-function
implements the straighforward mapping :math:`(s,a)\mapsto Q(s,a)`, while a
type-II Q-function implements :math:`s\mapsto Q(s,.)`.



Reference
---------

.. toctree::
    :maxdepth: 2
    :glob:

    state
    state_action
