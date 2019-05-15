keras-gym
==========

*Plug-n-play Reinforcement Learning in Python*


.. raw:: html

    <div align="center">
    <video autoplay loop muted id="cartpole">
      <source src="_static/video/cartpole.mp4" type="video/mp4">
    </video></div>


Create simple, reproducible RL solutions with OpenAI gym environments and Keras
function approximators. Also, compatibility wrappers for scikit-learn are included.


Documentation
-------------

.. toctree::
    :maxdepth: 1

    notebooks/index
    value_functions/index
    policies/index
    misc/glossary
    misc/release_notes


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Example
-------

Check out the :doc:`notebooks/index` for examples.

Here's one of the examples from the notebooks, in which we solve the
`'CartPole-v0'` environment with the SARSA algorithm, using a linear function
approximator for Q(s, a):


.. code:: python

    import gym

    from keras_gym.preprocessing import DefaultPreprocessor
    from keras_gym.value_functions import LinearQTypeI
    from keras_gym.policies import EpsilonGreedy


    # env with preprocessing
    env = gym.make('Carpole-v0')
    env = DefaultPreprocessor(env)

    # value function and its derived policy
    Q = LinearQTypeI(env, lr=0.05, gamma=0.8, update_strategy='sarsa', bootstrap_n=1)
    policy = EpsilonGreedy(Q)

    # static parameters
    num_episodes = 200
    num_steps = env.spec.max_episode_steps

    # used for early stopping
    num_consecutive_successes = 0


    # train
    for ep in range(num_episodes):
        s = env.reset()
        epsilon = 0.1 if ep < 10 else 0.01

        for t in range(num_steps):
            a = policy(s, epsilon)
            s_next, r, done, info = env.step(a)

            Q.update(s, a, r, done)

            if done:
                if t == num_steps - 1:
                    num_consecutive_successes += 1
                    print("num_consecutive_successes: {}"
                          .format(num_consecutive_successes))
                else:
                    num_consecutive_successes = 0
                    print("failed after {} steps".format(t))
                break

            s = s_next

        if num_consecutive_successes == 10:
            break


    # run env one more time to render
    s = env.reset()
    env.render()

    for t in range(num_steps):

        a = policy(s, epsilon=0)
        s, r, done, info = env.step(a)
        env.render()

        if done:
            break

    env.close()

