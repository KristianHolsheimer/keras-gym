import gym
import numpy as np
import keras_gym as km
import datetime
from gym.envs.toy_text.frozen_lake import UP, DOWN, LEFT, RIGHT


# the MDP
actions = {LEFT: 'L', RIGHT: 'R', UP: 'U', DOWN: 'D'}
env = gym.make('FrozenLakeNonSlippery-v0')
tbdir = datetime.datetime.now().strftime('data/tensorboard/%Y%m%d_%H%M%S')
env = km.wrappers.TrainMonitor(env, tensorboard_dir=tbdir)


# show logs from TrainMonitor
km.enable_logging()


class LinearFunc(km.FunctionApproximator):
    """ linear function approximator (body only does one-hot encoding) """
    pass


# define function approximators
func = LinearFunc(env, lr=0.01)
pi = km.SoftmaxPolicy(func, entropy_beta=0.02)
v = km.V(func, gamma=0.9, bootstrap_n=1)
q1 = km.QTypeI(func, gamma=0.9, bootstrap_n=1)
q2 = km.QTypeI(func, gamma=0.9, bootstrap_n=1)

sac = km.SoftActorCritic(pi, v, q1, q2)


# static parameters
target_model_sync_period = 10
num_episodes = 250


# train
for ep in range(num_episodes):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=False)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if np.array_equal(s_next, s):
            r = -0.1

        sac.update(s, a, r, done)

        if env.T % target_model_sync_period == 0:
            pi.sync_target_model(tau=1.0)

        if done:
            break

        s = s_next


# run one more episode to inspect the final result
s = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # print individual action probabilities
    print("  v(s) = {:.3f}".format(v(s)))
    for i, p in enumerate(km.utils.softmax(sac.policy.dist_params(s))):
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))
    for i, q in enumerate(sac.q_func1(s)):
        print("  q1(s,{:s}) = {:.3f}".format(actions[i], q))
    for i, q in enumerate(sac.q_func2(s)):
        print("  q2(s,{:s}) = {:.3f}".format(actions[i], q))

    a = sac.policy.greedy(s)
    s, r, done, info = env.step(a)
    env.render()

    if done:
        break
