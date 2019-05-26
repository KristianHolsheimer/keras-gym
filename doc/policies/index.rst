.. automodule:: keras_gym.policies


Policies
========

In reinforcement learning, a policy can either be derived from a state-action
value function (Q-function) or it be a stand-alone object the bares no
reference to a value function. These two approaches are called *value-based*
and *policy-based* RL, respectively. The way we update our policies differs
quite a bit between the two approaches.

For value-based RL, we have algorithms like TD(0), Monte Carlo and everything
in between. The optimization problem that we use to update our function
approximator is typically ordinary least-squares regression (or Huber loss).

In policy-based RL, on the other hand, we update our function approximators
using direct policy gradient techniques. This makes the optimization problem
quite different from ordinary supervised learning.

Below you'll find the generic specification for all policy objects:


Value-Based Policies
--------------------

These policies are derived from a Q-function object. See example below:

.. code:: python

    import gym

    from keras_gym.value_functions import LinearQTypeI
    from keras_gym.policies import EpsilonGreedy
    from keras_gym.preprocessing import DefaultPreprocessor


    # env with preprocessing
    env = gym.make(...)
    env = DefaultPreprocessor(env)

    # use linear function approximator for Q(s,a)
    Q = LinearQTypeI(env, ...)
    policy = EpsilonGreedy(Q, epsilon=0.1)

    # get some dummy state observation
    s = env.reset()

    # draw an action, given state s
    a = policy(s)


Updateable Policies
-------------------

These policies can be updated directly by policy-gradient methods. See example
below, in which we implement the REINFORCE algorithm:

.. code:: python

    import gym

    from keras_gym.caching import MonteCarloCache
    from keras_gym.policies import LinearSoftmaxPolicy
    from keras_gym.preprocessing import DefaultPreprocessor

    # env with preprocessing
    env = gym.make(...)
    env = DefaultPreprocessor(env)

    # use linear function approximator for pi(a|s)
    policy = LinearSoftmaxPolicy(env, lr=0.1, ...)
    cache = MonteCarloCache(gamma=0.99)

    # initialize env
    s = env.reset()
    cache.reset()

    # run episodes
    for ep in range(num_episodes):
        s = env.reset()

        for t in range(num_steps):
            a = policy(s)
            s_next, r, done, info = env.step(a)
            cache.add(s, a, r, done)

            if done:
                # update at the end of the episode
                while cache:
                    s, a, g = cache.pop()
                    policy.update(s, a, g)

                break

            s = s_next

    env.close()


Actor-Critic
------------

An :class:`ActorCritic <keras_gym.policies.ActorCritic>` combines an
:term:`updateable policy` with a :doc:`value function
<../value_functions/index>`. For example, below we define an actor-critic with
linear function approximators:

.. code:: python

    import gym

    from keras_gym.policies import LinearSoftmaxPolicy, ActorCritic
    from keras_gym.value_functions import LinearV
    from keras_gym.preprocessing import DefaultPreprocessor

    # env with preprocessing
    env = gym.make(...)
    env = DefaultPreprocessor(env)

    # define actor-critic
    policy = LinearSoftmaxPolicy(env, lr=0.1, update_strategy='vanilla')
    V = LinearV(env, lr=0.1, gamma=0.9, bootstrap_n=1)
    actor_critic = ActorCritic(policy, V)

    # run episodes
    for ep in range(num_episodes):
        s = env.reset()

        for t in range(num_steps):
            a = policy(s)
            s_next, r, done, info = env.step(a)

            actor_critic.update(s, a, r, done)

            if done:
                break

            s = s_next

    env.close()


Special Policies
----------------

We've also got some special policies, which are policies that don't depend on
any learned function approximator. The two main examples that are available
right now are :class:`RandomPolicy <keras_gym.policies.RandomPolicy>` and
:class:`UserInputPolicy <keras_gym.policies.UserInputPolicy>`. The latter
allows you to pick the actions yourself as the episode runs.


Reference
---------

.. toctree::
    :maxdepth: 2
    :glob:

    base
    value_based
    updateable
    actor_critic
    special
