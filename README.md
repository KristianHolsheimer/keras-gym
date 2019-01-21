# scikit-gym
*plug-n-play reinforcement learning*


## Documentation

For the docs, go to [scikit-gym.readthedocs.io](https://scikit-gym.readthedocs.io/)


## Install

Install using pip:

```
$ pip install -U scikit-gym
```
or install from a fresh clone
```
$ git clone https://github.com/KristianHolsheimer/scikit-gym.git
$ pip install -e ./scikit-gym
```

## Simple example

```python

import gym
from skgym.value_functions import LinearQ
from skgym.policies import ValuePolicy
from skgym.algorithms import QLearning


env = gym.make('CartPole-v0')
q = LinearQ(env, power_t=0)  # uses sklearn's SGDRegressor
policy = ValuePolicy(q)
algo = QLearning(policy, gamma=0.75)


# use this for early-stopping
consecutive_successes = 0


for episode in range(200):
    s = env.reset()
    a = env.action_space.sample()
    epsilon = 0.5 if episode < 20 else 0.01

    for t in range(200):
        s_next, r, done, info = env.step(a)

        # simple TD(0) update
        algo.update(s, a, r, s_next)

        if done:
            if t == 199:
                consecutive_successes += 1
            else:
                consecutive_successes = 0
            break

        # prepare for next step
        s = s_next
        a = policy.epsilon_greedy(s, epsilon=epsilon)

    if consecutive_successes == 10:
        break


# render under final policy
s = env.reset()

for t in range(200):
    env.render()

    a = policy(s)
    s, r, done, info = env.step(a)

    if done:
        break

env.render()
env.close()
```


## TODO

* add support for continuous action spaces
* check whether the above example still works
* implement experience cache for MC and experience-replay type algorithms
