# keras-gym
*Plug-n-Play Reinforcement Learning in Python*


Create simple, reproducible RL solutions with Keras function approximators.


## Documentation

For the full documentation, go to [keras-gym.readthedocs.io](https://keras-gym.readthedocs.io/)


## Install

Install using pip:

```
$ pip install -U keras-gym
```
or install from a fresh clone
```
$ git clone https://github.com/KristianHolsheimer/keras-gym.git
$ pip install -e ./keras-gym
```

## Examples

Check out the [notebooks](notebooks/) for examples. These are also included in
the documentation:

* https://keras-gym.readthedocs.io/notebooks/



Here's one of the examples from the notebooks, in which we solve the
`'CartPole-v0'` environment with the SARSA algorithm, using a linear function
approximator for Q(s, a):


```python
import numpy as np
import gym

from keras_gym.value_functions import GenericQ
from keras_gym.policies import ValuePolicy
from keras_gym.algorithms import Sarsa

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
```

The last episode is rendered, which shows something like this:

![cartpole_video](doc/_static/img/cartpole.gif)


## TODO

* add support for continuous action spaces
* ~check whether the above example still works~
* ~implement experience cache for MC implement experience-replay type algorithms~
* implement sparse one-hot vectors
* implement `utils.feature_vector` for `gym.spaces.Dict` space.
* fix slow monte-carlo algorithm (standalone benchmark in notebook is faster)
