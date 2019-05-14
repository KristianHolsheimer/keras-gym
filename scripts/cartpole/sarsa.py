import gym

from keras_gym.preprocessing import DefaultPreprocessor
from keras_gym.value_functions import LinearQTypeI
from keras_gym.policies import EpsilonGreedy


# env with preprocessing
env = gym.make('CartPole-v0')
env = DefaultPreprocessor(env)

# value function and its derived policy
Q = LinearQTypeI(env, lr=0.02, momentum=0.9, gamma=0.8,
                 update_strategy='sarsa', bootstrap_n=1)
policy = EpsilonGreedy(Q)

# static parameters
num_episodes = 200
num_steps = env.spec.max_episode_steps

# used for early stopping
num_consecutive_successes = 0


# train
for ep in range(num_episodes):
    s = env.reset()
    policy.epsilon = 0.1 if ep < 10 else 0.01

    for t in range(num_steps):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        Q.update(s, a, r, done)

        if done:
            if t == num_steps - 1:
                num_consecutive_successes += 1
                print("num_consecutive_successes: {}"
                      .format(num_consecutive_successes))
            else:
                num_consecutive_successes = 0
                print("failed after {} steps".format(t))
            break

        s = s_next

    if num_consecutive_successes == 10:
        break


# run env one more time to render
s = env.reset()
env.render()
policy.epsilon = 0

for t in range(num_steps):

    a = policy(s)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break

env.close()
