# flake8: noqa
import gym
from .self_play import *


gym.envs.register(
    id='ConnectFour-v0',
    entry_point='keras_gym.envs.self_play:ConnectFourEnv',
)

gym.envs.register(
    id='FrozenLakeNonSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=20,
    reward_threshold=0.99,
)

# clear gym from namespace
del gym