# keras-gym
*Plug-n-Play Reinforcement Learning in Python*


Create simple, reproducible RL solutions with Keras function approximators.


## Documentation

For the full documentation, go to
[keras-gym.readthedocs.io](https://keras-gym.readthedocs.io/)


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
import gym

from keras_gym.preprocessing import DefaultPreprocessor
from keras_gym.value_functions import LinearQTypeI
from keras_gym.policies import EpsilonGreedy


# env with preprocessing
env = gym.make('CartPole-v0')
env = DefaultPreprocessor(env)

# value function and its derived policy
Q = LinearQTypeI(env, update_strategy='sarsa', bootstrap_n=1, gamma=0.8,
                 interaction='elementwise_quadratic', lr=0.02, momentum=0.9)
policy = EpsilonGreedy(Q)

# static parameters
num_episodes = 200
num_steps = env.spec.max_episode_steps

# used for early stopping
num_consecutive_successes = 0


# train
for ep in range(num_episodes):
    s = env.reset()
    policy.epsilon = 0.1 if ep < 10 else 0.01

    for t in range(num_steps):
        a = policy(s)
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
policy.epsilon = 0

for t in range(num_steps):

    a = policy(s)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break

env.close()

```

The last episode is rendered, which shows something like this:

![cartpole_video](doc/_static/img/cartpole.gif)

