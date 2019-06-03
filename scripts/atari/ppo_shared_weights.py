import os
import logging
import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.utils import TrainMonitor, generate_gif
from keras_gym.policies import AtariActorCritic
from keras_gym.caching import ExperienceReplayBuffer


logging.basicConfig(level=logging.INFO)


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=3)
env = TrainMonitor(env)


# actor-critic with shared weights between pi(a|s) and V(s)
actor_critic = AtariActorCritic(
    env,
    gamma=0.99,
    bootstrap_n=10,
    update_strategy='ppo',
    ppo_clipping=0.2,
    entropy_bonus=0.01,
    lr=0.00025)

V = actor_critic.value_function
pi = actor_critic.policy


# we'll use this to temporarily store our experience
buffer = ExperienceReplayBuffer.from_value_function(
    V, capacity=256, batch_size=64)


# run episodes
while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=True)  # target_model == pi_old
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)

        if len(buffer) >= buffer.capacity:
            # use 4 epochs per round
            num_batches = int(4 * buffer.capacity / buffer.batch_size)
            for _ in range(num_batches):
                actor_critic.batch_update(*buffer.sample())
            buffer.clear()

            # soft update (tau=1 would be a hard update)
            actor_critic.sync_target_model(tau=0.1)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.ep % 50 == 0:
        os.makedirs('./data/ppo/gifs/', exist_ok=True)
        generate_gif(
            env=env,
            policy=pi,
            filepath='./data/ppo/gifs/ep{:06d}.gif'.format(env.ep),
            resize_to=(320, 420))
