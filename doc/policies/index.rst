Policies
========

In reinforcement learning (RL), a policy can either be derived from a
:term:`state-action value function <type-I state-action value function>` or it
be learned directly as an :term:`updateable policy`. These two
approaches are called *value-based* and *policy-based* RL, respectively. The
way we update our policies differs quite a bit between the two approaches.

For value-based RL, we have algorithms like TD(0), Monte Carlo and everything
in between. The optimization problem that we use to update our function
approximator is typically ordinary least-squares regression (or Huber loss).

In policy-based RL, on the other hand, we update our function approximators
using direct policy gradient techniques. This makes the optimization problem
quite different from ordinary supervised learning.

Below we list all policy objects provided by **keras-gym**.



Updateable Policies and Actor-Critics
-------------------------------------

For :term:`updateable policies <updateable policy>` have a look at the relevant
:term:`function approximator` section:

- :doc:`Function Approximators <../function_approximators/index>`.



Value-Based Policies
--------------------

These policies are derived from a Q-function object. See example below:

.. code:: python

    import gym
    import keras_gym as km

    # the cart-pole MDP
    env = gym.make(...)

    # use linear function approximator for q(s,a)
    func = km.predefined.LinearFunctionApproximator(env, lr=0.01)
    q = km.Q(func, update_strategy='q_learning')
    pi = EpsilonGreedy(q, epsilon=0.1)

    # get some dummy state observation
    s = env.reset()

    # draw an action, given state s
    a = pi(s)


Special Policies
----------------

We've also got some special policies, which are policies that don't depend on
any learned function approximator. The two main examples that are available
right now are :class:`RandomPolicy <keras_gym.policies.RandomPolicy>` and
:class:`UserInputPolicy <keras_gym.policies.UserInputPolicy>`. The latter
allows you to pick the actions yourself as the episode runs.


Objects
-------

.. toctree::

    value_based
    special
