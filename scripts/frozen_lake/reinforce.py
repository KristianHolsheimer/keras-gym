import keras_gym as km
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, UP, DOWN, LEFT, RIGHT


# env with preprocessing
actions = {LEFT: 'L', RIGHT: 'R', UP: 'U', DOWN: 'D'}
env = FrozenLakeEnv(is_slippery=False)
env = km.utils.TrainMonitor(env)


class LinearFunctionApproximator(km.FunctionApproximator):
    def body(self, S, variable_scope):
        one_hot = keras.layers.Lambda(lambda x: K.one_hot(x, 16))
        return one_hot(S)


# define function approximators
func = LinearFunctionApproximator(env, lr=0.01)
pi = km.SoftmaxPolicy(func, update_strategy='vanilla')
cache = km.MonteCarloCache(gamma=0.99)


# static parameters
num_episodes = 1000
num_steps = 30


# train
for ep in range(num_episodes):
    s = env.reset()
    cache.reset()

    for t in range(num_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if np.array_equal(s_next, s):
            r = -0.1

        cache.add(s, a, r, done)

        if done:
            while cache:
                S, A, G = cache.pop()
                pi.batch_update(S, A, G)
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(num_steps):

    # print individual action probabilities
    for i, p in enumerate(pi.proba(s)):
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
