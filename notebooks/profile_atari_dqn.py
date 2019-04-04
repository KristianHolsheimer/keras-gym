#!/usr/bin/env python3
import os
import glob
import time
import re
import sys

import numpy as np
import gym

from keras_gym.utils import ArrayDeque
from keras_gym.value_functions.predefined.atari import AtariDQN
from keras_gym.policies import ValueBasedPolicy
from keras_gym.algorithms import QLearning

env = gym.make('MsPacman-v0')
Q = AtariDQN(env, target_model_sync_period=100, target_model_sync_tau=0.1)
policy = ValueBasedPolicy(Q, boltzmann_temperature=0.01)
algo = QLearning(
    Q, gamma=0.99, experience_cache_size=250000,
    experience_replay_batch_size=32)


# make sure dir exists
os.makedirs('data/checkpoints', exist_ok=True)

# load weights
if glob.glob('data/checkpoints/*'):
    weights_filepath = max(glob.glob('data/checkpoints/*'))
    print(f"loading model weights from: {weights_filepath}")
    weights = list(np.load(weights_filepath).values())
    Q.model.set_weights(weights)
    if Q.target_model is not None:
        Q.target_model.set_weights(weights)
    start_ep = 1 + int(re.match(
        r'^data/checkpoints/ep(\d{4}).npz$', weights_filepath).groups()[0])
else:
    print(f"couldn't load weights, will train from scratch")
    start_ep = 1


# keep track of last 100 episode avg reward
G_100 = ArrayDeque(shape=[], maxlen=100, overflow='cycle')
t_100 = ArrayDeque(shape=[], maxlen=100, overflow='cycle')
dt_100 = ArrayDeque(shape=[], maxlen=100, overflow='cycle')


def run(num_episodes, render=False):
    T = 0
    for ep in range(start_ep, start_ep + num_episodes):
        G = 0.0
        dt = time.time()
        s = env.reset()
        if render:
            env.render()
            time.sleep(0.02)

        for t in range(1, 1 + env._max_episode_steps):
            a = policy.epsilon_greedy(s, epsilon=0.1)
            s_next, r, done, info = env.step(a)
            if render:
                env.render()
                time.sleep(0.02)

            # global counters
            T += 1
            G += r

            algo.update(s, a, r, s_next, done)

            if done:
                break

            s = s_next

        dt = time.time() - dt

        G_100.append(G)
        t_100.append(t)
        dt_100.append(dt)
        avg_r_100 = G_100.array.sum() / t_100.array.sum()
        avg_G_100 = G_100.array.mean()
        avg_t_100 = t_100.array.mean()
        avg_dt_100 = dt_100.array.sum() / t_100.array.sum()
        print(
            f"ep = {ep}, T={T}, "
            f"t = {t:d}(1) / {avg_t_100:.01f}(100), "
            f"G = {G:.00f}(1) / {avg_G_100:.01f}(100), "
            f"avg(r) = {G/t:.03f}(1) / {avg_r_100:.03f}(100), "
            f"dt_ms = {1e3*dt/t:.02f}(1) / {1e3*avg_dt_100:.02f}(100), "
        )
        sys.stdout.flush()

        np.savez_compressed(
            'data/checkpoints/ep{:04d}.npz'.format(ep), *Q.model.get_weights())


if __name__ == '__main__':
    run(100000, render=False)
