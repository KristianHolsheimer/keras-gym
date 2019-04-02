import os
import glob
import time
import re

import numpy as np
import gym

from keras_gym.utils import ArrayDeque
from keras_gym.value_functions.predefined.atari import AtariDQN
from keras_gym.policies import ValueBasedPolicy
from keras_gym.algorithms import QLearning

env = gym.make('MsPacman-v0')
Q = AtariDQN(env, target_model_sync_period=100, target_model_sync_tau=0.1)
policy = ValueBasedPolicy(Q, boltzmann_temperature=0.1)
algo = QLearning(Q, gamma=0.99, experience_cache_size=80000, experience_replay_batch_size=32)


# make sure dir exists
os.makedirs('data/checkpoints', exist_ok=True)

# load weights
if glob.glob('data/checkpoints/*'):
    weights_filepath = max(glob.glob('data/checkpoints/*'))
    print(f"loading model weights from: {weights_filepath}")
    weights = list(np.load(weights_filepath).values())
    Q.model.set_weights(weights)
    start_ep = 1 + int(re.match(
        '^data/checkpoints/ep(\d{4}).npz$', weights_filepath).groups()[0])
else:
    print(f"couldn't load weights, will train from scratch")
    start_ep = 1


# keep track of last 100 episode avg reward
G_100 = ArrayDeque(shape=[], maxlen=100, overflow='cycle')


def run(num_episodes, render=False):
    T = 0
    G_all = 0.0
    for ep in range(start_ep, start_ep + num_episodes):
        G_ep = 0
        s = env.reset()
        if render:
            env.render()

        for t in range(1, 1 + env._max_episode_steps):
            a = policy.epsilon_greedy(s, epsilon=np.clip(1 - T / 100000., 0.1, 1.0))
            s_next, r, done, info = env.step(a)
            if render:
                env.render()
                time.sleep(0.01)

            # global counters
            T += 1
            G_ep += r

            # algo.update(s, a, r, s_next, done)

            if done:
                break

            s = s_next

        G_all += G_ep
        G_100.append(G_ep / t)
        print(f"ep={ep}, t={t}, T={T}, avg_r_ep={G_ep/t}, "
              f"avg_r_100={G_100.array.mean()}, avg_r_all={G_all/T}")

        np.savez_compressed(
            'data/checkpoints/ep{:04d}.npz'.format(ep), *Q.model.get_weights())


if __name__ == '__main__':
    run(1000, render=False)
