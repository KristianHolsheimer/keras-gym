Function Approximators
======================

The central object object in this package is the
:class:`keras_gym.FunctionApproximator`, which provides an interface between a
gym-type environment and function approximators like :term:`value functions
<state value function>` and :term:`updateable policies <updateable policy>`.



FunctionApproximator class
--------------------------

The way we would define a function approximator is by specifying a
:term:`body`. For instance, the example below specifies a simple multi-layer
perceptron:

.. code:: python

    import gym
    import keras_gym as km
    from tensorflow import keras


    class MLP(km.FunctionApproximator):
        """ multi-layer perceptron with one hidden layer """
        def body(self, S, variable_scope):
            X = keras.layers.Flatten()(S)
            X = keras.layers.Dense(units=4, name=(variable_scope + '/hidden'))(X)
            return X


    # environment
    env = gym.make(...)

    # value function and its derived policy
    function_approximator = MLP(env, lr=0.01)


This ``function_approximator`` can now be used to construct a value function or
updateable policy, which we cover in the remainder of this page.



Predefined Function Approximators
---------------------------------

Although it's pretty easy to create a custom function approximator,
**keras-gym** also provides some predefined function approximators. They are
listed :doc:`here <predefined>`.



Value Functions
---------------

Value functions estimate the expected (discounted) sum of future rewards. For
instance, :term:`state value functions <state value function>` are defined as:

.. math::

    v(s)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots\ \Big|\ S_t=s
    \right\}

Here, the :math:`R` are the individual rewards we receive from the Markov
Decision Process (MDP) at each time step.

In **keras-gym** we define a state value functions as follows:

.. code:: python

    v = km.V(function_approximator, gamma=0.9, bootstrap_n=1)

The `function_approximator <FunctionApproximator objects>`_ is discussed above.
The other arguments set the discount factor :math:`\gamma\in[0,1]` and the
number of steps over which to bootstrap.


Similar to state value functions, we can also define state-action value
functions:

.. math::

    q(s, a)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots\ \Big|\ S_t=s, A_t=a
    \right\}

**keras-gym** provides two distinct ways to define such a Q-function, which are
refered to as :term:`type-I <type-I state-action value function>` and
:term:`type-II <type-II state-action value function>` Q-functions. The
difference between the two is in how the function approximator models the
Q-function. A type-I Q-function models the Q-function as :math:`(s, a)\mapsto
q(s, a)\in\mathbb{R}`, whereas a type-II Q-function models it as
:math:`s\mapsto q(s,.)\in\mathbb{R}^n`. Here, :math:`n` is the number of
actions, which means that this is only well-defined for discrete action spaces.

In **keras-gym** we define a type-I Q-function as follows:

.. code:: python

    q = km.QTypeI(function_approximator, update_strategy='sarsa')

and similarly for type-II:

.. code:: python

    q = km.QTypeII(function_approximator, update_strategy='sarsa')

The ``update_strategy`` argument specifies our bootstrapping target. Available
choices are ``'sarsa'``, ``'q_learning'`` and ``'double_q_learning'``.

The main reason for using a Q-function is for value-based control. In other
words, we typically want to derive a policy from the Q-function. This is pretty
straightforward too:

.. code:: python

    pi = km.EpsilonGreedy(q, epsilon=0.1)

    # the epsilon parameter may be updated dynamically
    pi.set_epsilon(0.25)



Updateable Policies
-------------------

Besides value-based control in which we derive a policy from a Q-function, we
can also do policy-based control. In policy-based methods we learn a policy
directly as a probability distribution over the space of actions
:math:`\pi(a|s)`.

The updateable policies for discrete action spaces are known as softmax
policies:

.. math::

    \pi(a|s)\ =\ \frac{\exp z(s,a)}{\sum_{a'}\exp z(s,a')}

where the logits are defined over the real line :math:`z(s,a)\in\mathbb{R}`.

In **keras-gym** we define a softmax policy as follows:

.. code:: python

    pi = km.SoftmaxPolicy(function_approximator, update_strategy='vanilla')

Similar to Q-functions, we can pick different update strategies. Available
options for policies are ``'vanilla'``, ``'ppo'`` and ``'cross_entropy'``.
These specify the objective function used in our policy updates.


Actor-Critics
-------------

It's often useful to combine a policy with a value function into what is called
an :term:`actor-critic`. The value function (critic) can be used to aid the
update procedure for the policy (actor). The **keras-gym** package provides two
kinds of actor-critic classes: :class:`ActorCritic <keras_gym.ActorCritic>` and
:class:`ConjointActorCritic <keras_gym.ConjointActorCritic>`. The former is
basically just a wrapper that combines a policy with a separate value function:

.. code:: python

    # separate policy and value function
    pi = km.SoftmaxPolicy(function_approximator, update_strategy='vanilla')
    v = km.V(function_approximator, gamma=0.9, bootstrap_n=1)

    # combine them into a single actor-critic
    actor_critic = km.ActorCritic(pi, v)

This works fine for relatively small function approximators. For very large
ones, however, we might be better off sharing part of the computation graph
between policy and value function. The part of the function approximator that's
shared is the part defined in the :func:`body
<keras_gym.FunctionApproximator.body>` method. The class that implements this
configuration is :class:`ConjointActorCritic <keras_gym.ConjointActorCritic>`:

.. code:: python

    # a single actor-critic
    actor_critic = km.ConjointActorCritic(
        function_approximator, update_strategy='vanilla', gamma=0.9, bootstrap_n=1)

    # the policy and value function are still separately accessible
    pi = actor_critic.policy
    v = actor_critic.value_function


Objects
-------

.. toctree::

    generic
    predefined
    value_functions
    updateable_policies
    actor_critics
