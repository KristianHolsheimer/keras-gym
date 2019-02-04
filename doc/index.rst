scikit-gym
==========

*Plug-n-play Reinforcement Learning in Python*


.. raw:: html

    <div align="center">
    <video autoplay loop muted id="cartpole">
      <source src="_static/video/cartpole.mp4" type="video/mp4">
    </video></div>


Create simple, reproducible RL solutions with scikit-learn style function
approximators.


Documentation
-------------

.. toctree::
    :maxdepth: 1

    notebooks/index
    value_functions/index
    policies/index
    algorithms/index
    environments/index
    misc/utils
    misc/about
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

    import numpy as np
    import gym

    from skgym.value_functions import GenericQ
    from skgym.policies import ValuePolicy
    from skgym.algorithms import Sarsa

    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import FunctionTransformer


    # the Gym environment
    env = gym.make('CartPole-v0')


    # define sklearn model for approximating Q-function
    regressor = SGDRegressor(eta0=0.05, learning_rate='constant')
    transformer = FunctionTransformer(
        lambda x: np.hstack((x, x ** 2)), validate=False)


    # define Q, its induced policy and update algorithm
    Q = GenericQ(env, regressor, transformer)
    policy = ValuePolicy(Q)
    algo = Sarsa(Q, gamma=0.8)


    # number of iterations
    num_episodes = 200
    max_episode_steps = env._max_episode_steps


    # used for early stopping
    num_consecutive_successes = 0


    # run the episodes
    for episode in range(1, num_episodes + 1):
        last_episode = episode == num_episodes or num_consecutive_successes == 9

        # init
        s = env.reset()
        a = env.action_space.sample()

        # exploration schedule
        epsilon = 0.1 if episode < 10 else 0.01

        if last_episode:
            epsilon = 0  # no more exploration
            env.render()

        # run episode
        for t in range(1, max_episode_steps + 1):
            s_next, r, done, info = env.step(a)
            a_next = policy.epsilon_greedy(s, epsilon)

            # update or render
            if not last_episode:
                algo.update(s, a, r, s_next, a_next)
            else:
                env.render()

            # keep track of consecutive successes
            if done:
                if t == max_episode_steps:
                    num_consecutive_successes += 1
                    print(f"episode = {episode}, num_consecutive_successes = {num_consecutive_successes}")
                else:
                    num_consecutive_successes = 0
                    print(f"episode = {episode}, failed after {t} steps")
                break

            # prepare for next step
            s, a = s_next, a_next


        if last_episode:
            break


    env.close()
