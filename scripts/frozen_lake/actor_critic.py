import logging

import numpy as np
import keras_gym as km
from tensorflow import keras
from tensorflow.keras import backend as K
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, LEFT, RIGHT, UP, DOWN


logging.basicConfig(level=logging.INFO)


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
pi = km.SoftmaxPolicy(func, update_strategy='ppo')
v = km.V(func, gamma=0.9, bootstrap_n=1)


# combine into one actor-critic
actor_critic = km.ActorCritic(pi, v)


# static parameters
target_model_sync_period = 20
num_episodes = 250
num_steps = 30


# train
for ep in range(num_episodes):
    s = env.reset()

    for t in range(num_steps):
        a = pi(s, use_target_model=True)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if np.array_equal(s_next, s):
            r = -0.1

        actor_critic.update(s, a, r, done)

        if env.T % target_model_sync_period == 0:
            pi.sync_target_model(tau=1.0)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(num_steps):

    # print individual action probabilities
    print("  V(s) = {:.3f}".format(v(s)))
    for i, p in enumerate(pi.proba(s)):
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break
