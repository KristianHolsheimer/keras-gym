import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, UP, DOWN, LEFT, RIGHT

from keras_gym.preprocessing import DefaultPreprocessor
from keras_gym.value_functions import LinearV
from keras_gym.policies import LinearSoftmaxPolicy
from keras_gym.caching import MonteCarloCache


# env with preprocessing
env = FrozenLakeEnv(is_slippery=False)
env = DefaultPreprocessor(env)
actions = {LEFT: 'L', RIGHT: 'R', UP: 'U', DOWN: 'D'}

# updateable policy
V = LinearV(env, lr=0.1, bootstrap_n=1)
policy = LinearSoftmaxPolicy(env, lr=0.1)
cache = MonteCarloCache(gamma=0.99)


# static parameters
num_episodes = 1000
num_steps = 30


# train
for ep in range(num_episodes):
    s = env.reset()
    cache.reset()

    for t in range(num_steps):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if np.array_equal(s_next, s):
            r = -0.1

        V.update(s, r, done)

        cache.add(s, a, r, done)

        if done:
            while cache:
                S, A, G = cache.pop()
                policy.batch_update(S, A, G)
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(num_steps):

    # print individual action probabilities
    print("  V(s) = {:.3f}".format(V(s)))
    for i, p in enumerate(policy.proba(s)):
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))

    a = policy.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
