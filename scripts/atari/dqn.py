import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.value_functions import AtariQ
from keras_gym.policies import EpsilonGreedy  # or BoltzmannPolicy
from keras_gym.caching import ExperienceReplayBuffer


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env)
env = FrameStacker(env)


# value function
Q = AtariQ(env, lr=0.00025, gamma=0.99, bootstrap_n=1)
buffer = ExperienceReplayBuffer(Q, capacity=1000000, batch_size=32)
policy = EpsilonGreedy(Q)


# exploration schedule
def epsilon(T):
    """ piecewise linear annealing """
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


# global step counter
T = 0
T_sync = 0


for ep in range(num_episodes):
    s = env.reset()
    G = 0  # to accumulate return

    for t in range(num_steps):
        a = policy(s, epsilon=epsilon(T))
        s_next, r, done, info = env.step(s)

        # counters
        G += r
        T += 1
        T_sync += 1

        buffer.add(s, a, r, done, info, ep)

        if T > buffer_warmup_period:
            Q.batch_update(*buffer.sample())

        if T_sync % target_model_sync_period == 0:
            Q.sync_target_model()
            T_sync = 0

        if done:
            print("ep: {}, T: {}, G: {}, t: {}".format(ep, T, G, t))
            break

        s = s_next
