from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from keras_gym.preprocessing import DefaultPreprocessor
from keras_gym.policies import Policy
from keras_gym.caching import MonteCarlo


# env with preprocessing
env = FrozenLakeEnv(is_slippery=False)
env = DefaultPreprocessor(env)


# updateable policy
policy = Policy(env, lr=0.1)
buffer = MonteCarlo(gamma=0.9)


# static parameters
num_episodes = 50
num_steps = 30


# train
for ep in range(num_episodes):
    s = env.reset()

    for t in range(num_steps):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r)

        if done:
            policy.batch_update(*buffer.flush())
            break


# run env one more time to render
s = env.reset()
env.render()

for t in range(num_steps):

    a = policy(s, epsilon=0)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break
