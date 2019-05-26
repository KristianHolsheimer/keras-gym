import os
import logging
import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.utils import TrainMonitor, generate_gif
from keras_gym.value_functions import AtariQ
from keras_gym.policies import EpsilonGreedy  # or BoltzmannPolicy
from keras_gym.caching import ExperienceReplayBuffer


logging.basicConfig(level=logging.INFO)


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=3)
env = TrainMonitor(env)


# value function
Q = AtariQ(env, lr=0.00025, gamma=0.99, bootstrap_n=1)
buffer = ExperienceReplayBuffer.from_q_function(Q, capacity=1e6, batch_size=32)
policy = EpsilonGreedy(Q)


# exploration schedule
def epsilon(T):
    """ stepwise linear annealing """
    M = 1000000
    if T < M:
        return 1 - 0.9 * T / M
    if T < 2 * M:
        return 0.1 - 0.09 * (T - M) / M
    return 0.01


# static parameters
num_episodes = 3000000
num_steps = env.spec.max_episode_steps
buffer_warmup_period = 50000
target_model_sync_period = 10000


for _ in range(num_episodes):
    if env.ep % 10 == 0 and env.T > buffer_warmup_period:
        os.makedirs('./data/gifs/', exist_ok=True)
        generate_gif(
            env=env,
            policy=policy.set_epsilon(0.01),
            filepath='./data/gifs/ep{:06d}.gif'.format(env.ep),
            resize_to=(320, 420))

    s = env.reset()

    for t in range(num_steps):
        policy.epsilon = epsilon(env.T)
        a = policy(s)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)

        if env.T > buffer_warmup_period:
            Q.batch_update(*buffer.sample())

        if env.T % target_model_sync_period == 0:
            Q.sync_target_model()

        if done:
            break

        s = s_next
