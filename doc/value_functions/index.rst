.. automodule:: keras_gym.value_functions


Value Functions
===============

Value functions play a cetral role in reinforcement learning. For instance, in
value-based RL, we derive a policy from a state-action value function
:math:`Q(s,a)`. In other situations such as actor-critic type approaches we
make heavy use of value functions to estimate the advantage function
:math:`\mathcal{A}(s,a) = Q(s,a) - V(s)`.

Predefined Value Functions
--------------------------

Here we provide some value functions with predefined function approximators.
For instance, here's how you might use a value function with a linear function
approximator:


.. code:: python

    import gym

    from keras_gym.value_functions import LinearQ
    from keras_gym.policies import ValuePolicy
    from keras_gym.algorithms import Sarsa


    env = gym.make(...)

    # define Q, its induced policy and update algorithm
    Q = LinearQ(env, lr=0.08, interaction='elementwise_quadratic')
    policy = ValuePolicy(Q)
    algo = Sarsa(Q, gamma=0.8)

    # the rest of your code
    ...


Generic Value Functions
-----------------------

We also provide a generic interface if the predefined value functions don't
fit your specific needs. Here's an example that closely resembles the LinearQ
example above:


.. code:: python

    import gym

    from tensorflow import keras
    from tensorflow.keras import backend as K

    from keras_gym.value_functions import GenericQ
    from keras_gym.policies import ValuePolicy
    from keras_gym.algorithms import QLearning


    env = gym.make(...)

    # custom function apprixmator (linear regression)
    model = keras.Sequential(layers=[
        keras.layers.Lambda(lambda x: K.concatenate([x, x ** 2])),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.5),
        loss=keras.losses.mean_squared_error)


    # define Q, its induced policy and update algorithm
    Q = GenericQ(env, model)
    policy = ValuePolicy(Q)
    algo = QLearning(Q, gamma=0.8)

    # the rest of your code
    ...


References
----------

.. toctree::
    :maxdepth: 2
    :glob:

    predefined
    generic
