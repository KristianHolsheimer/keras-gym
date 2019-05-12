from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from keras_gym.preprocessing import DefaultPreprocessor
from keras_gym.policies import Policy, ActorCritic
from keras_gym.value_functions import LinearV


# env with preprocessing
env = FrozenLakeEnv(is_slippery=False)
env = DefaultPreprocessor(env)


# updateable policy
policy = Policy(env, lr=0.1, update_strategy='vanilla')
V = LinearV(env, lr=0.1, gamma=0.9, bootstrap_n=1)
actor_critic = ActorCritic(policy, V)


# static parameters
num_episodes = 50
num_steps = 30


# train
for ep in range(num_episodes):
    s = env.reset()

    for t in range(num_steps):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        actor_critic.update(s, a, r, done)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(num_steps):

    a = policy(s, epsilon=0)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break
