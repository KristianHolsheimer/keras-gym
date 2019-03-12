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
approximator is typically plain-vanilla least-squares regression.

In policy-based RL, on the other hand, we update our function approximators
using direct policy gradient techniques. This makes the optimization problem
quite different from ordinary supervised learning. In the `Generic Policies`_
section below you'll find in the code example, which illustrates the non-
standard nature of this optimization problem. In particular, the definition of
loss function :class:`SoftmaxPolicyLossWithLogits
<keras_gym.losses.SoftmaxPolicyLossWithLogits>` shows that we really require
the full flexibility of an auto-diff package like keras in order to make it
work.


Value-Based Policies
--------------------

These are policies based on a state-action value function :math:`Q(s,a)`. These
policies are updated implicitly by updating the underlying value function, see
example below:

.. code:: python

    import gym

    from keras_gym.value_functions import LinearQ
    from keras_gym.policies import ValuePolicy
    from keras_gym.algorithms import QLearning

    env = gym.make(...)

    # use linear function approximator for Q
    Q = LinearQ(env, lr=0.1)
    policy = ValuePolicy(Q)
    algo = QLearning(Q)

    # get some dummy state observation
    s = env.reset()

    # draw an action, given state s
    a = policy.epsilon_greedy(s, epsilon=0.1)

    ...



Predefined Policies
-------------------

These are policies that are directly updateable through policy-gradient
updates. They are specific implementations of generic policies (see section
below) and they're included in this package for convenience.


.. code:: python

    import gym
    from tensorflow import keras

    from keras_gym.policies import LinearSoftmaxPolicy
    from keras_gym.algorithms import Reinforce


    env = gym.make(...)

    # the policy and its updating algorithm
    policy = LinearSoftmaxPolicy(env, model)
    algo = Reinforce(policy, gamma=1.0)

    # rest of the code
    ...



Generic Policies
----------------

These policies are updateable through policy-gradient updates. They're generic
in the sense that the specific function approximator (`keras.Model` object)
must be supplied by the user.

Here's an example of how to construct a custom function approximator. Note that
this example is gives us a model that's basically the same as the predefined
:class:`LinearSoftmaxPolicy <keras_gym.policies.LinearSoftmaxPolicy>`.

.. code:: python

    import gym
    from tensorflow import keras

    from keras_gym.policies import GenericSoftmaxPolicy
    from keras_gym.losses import SoftmaxPolicyLossWithLogits
    from keras_gym.algorithms import Reinforce


    def create_model(num_features, num_actions):

        # inputs
        X = keras.Input(shape=[num_features])
        advantages = keras.Input(shape=[1])

        # computation graph
        dense = keras.layers.Dense(num_actions, kernel_initializer='zeros')
        logits = dense(X)

        # loss
        loss_function = SoftmaxPolicyLossWithLogits(advantages)

        # the final model
        model = keras.Model(inputs=[X, advantages], outputs=logits)
        model.compile(
            loss=loss_function,
            optimizer=keras.optimizers.SGD(lr=0.1))

        return model


    # some environment with discrete action space
    env = gym.make(...)

    # create a dummy feature vector to figure out the input dimension
    x = feature_vector(env.observation_space.sample(), env.observation_space)

    # function approximator
    model = create_model(num_features=x.size, num_actions=env.action_space.n)

    # the policy and its updating algorithm
    policy = GenericSoftmaxPolicy(env, model)
    algo = Reinforce(policy, gamma=1.0)

    # rest of the code
    ...


Reference
---------

.. toctree::
    :maxdepth: 2
    :glob:

    value_based
    predefined
    generic
