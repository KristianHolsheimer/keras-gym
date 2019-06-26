Function Approximators
======================

The central object object in this package is the
:class:`keras_gym.FunctionApproximator`, which provides an interface between a
gym-type environment and function approximators like :term:`value functions
<state value function>` and :term:`updateable policies`.

The way we would define a function approximator is by specifying a
:term:`body`. For instance, the example below specifies a simple multi-layer
perceptron:

.. code:: python

    import gym
    import keras_gym as km
    from tensorflow.keras.layers import Dense


    class MLP(km.FunctionApproximator):
        def body(self, S, variable_scope):
            X = Dense(units=9, name=(variable_scope + '/dense1'))(S)
            X = Dense(units=3, name=(variable_scope + '/dense2'))(X)
            return X


    # create environment
    env = gym.make('CartPole-v0')

    # define function approximators
    mlp = MLP(env, lr=0.01)
    v = km.V(mlp, gamma=0.9, bootstrap_n=1)
    pi = km.Policy(mlp, update_strategy='ppo')

    # combine into one actor-critic
    actor_critic = km.ActorCritic(pi, v)

    # run env
    for episode in range(500):
        s = env.reset()

        for t in range(env.spec.max_time_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            actor_critic.update(s, a, r, done)

            if done:
                break

            s = s_next
