import os
import logging
import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.utils import TrainMonitor, generate_gif
from keras_gym.value_functions import AtariV
from keras_gym.policies import AtariPolicy, ActorCritic
from keras_gym.caching import ExperienceReplayBuffer


logging.basicConfig(level=logging.INFO)


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=3)
env = TrainMonitor(env)


# value function
V = AtariV(
    env, lr=0.00025, gamma=0.99, bootstrap_n=10,
    bootstrap_with_target_model=True)
buffer = ExperienceReplayBuffer.from_value_function(
    V, capacity=256, batch_size=64)  # capacity is 'T' from Algo 1 [PPO paper]
policy = AtariPolicy(env, update_strategy='ppo', lr=0.00025)
actor_critic = ActorCritic(policy, V)


# static parameters
num_episodes = 3000000
num_steps = env.spec.max_episode_steps
num_epochs_per_agent_round = 4
num_batches_per_agent_round = int(
    num_epochs_per_agent_round * buffer.capacity / buffer.batch_size)


for _ in range(num_episodes):
    if env.ep % 50 == 0:
        os.makedirs('./data/ppo/gifs/', exist_ok=True)
        generate_gif(
            env=env,
            policy=policy,
            filepath='./data/ppo/gifs/ep{:06d}.gif'.format(env.ep),
            resize_to=(320, 420))

    s = env.reset()

    for t in range(num_steps):
        a = policy(s, use_target_model=True)  # target_model == pi(theta_old)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)

        if len(buffer) >= buffer.capacity:
            for _ in range(num_batches_per_agent_round):
                actor_critic.batch_update(*buffer.sample())
            buffer.clear()
            actor_critic.sync_target_model(tau=0.1)

        if done:
            break

        s = s_next
