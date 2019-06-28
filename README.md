# keras-gym
*Plug-n-Play Reinforcement Learning in Python*


![cartpole_video](doc/_static/img/cartpole.gif)

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

To get started, check out the [notebooks](notebooks/) for examples. These are
also included in the documentation:

* https://keras-gym.readthedocs.io/notebooks/



Here's one of the examples from the notebooks, in which we solve the
`CartPole-v0` environment with the SARSA algorithm, using a simple
multi-layer perceptron (MLP) with one hidden layer as our Q-function
approximator:


```python
import gym
import keras_gym as km
from tensorflow import keras

# the cart-pole MDP
env = gym.make('CartPole-v0')


class MLP(km.FunctionApproximator):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S, variable_scope):
        X = keras.layers.Flatten()(S)
        X = keras.layers.Dense(units=4, name=(variable_scope + '/hidden'))(X)
        return X


# value function and its derived policy pi(a|s)
func = MLP(env, lr=0.01)
q = km.QTypeI(func, update_strategy='sarsa')
pi = km.EpsilonGreedy(q)


# used for early stopping
num_consecutive_successes = 0


# train
for episode in range(200):
    s = env.reset()
    pi.epsilon = 0.1 if episode < 10 else 0.01

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        q.update(s, a, r, done)

        if done:
            if t == env.spec.max_episode_steps - 1:
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
pi.epsilon = 0

for t in range(env.spec.max_episode_steps):

    a = pi(s)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break

env.close()

```

The last episode is rendered, which shows something like the gif at the top of
this page.

