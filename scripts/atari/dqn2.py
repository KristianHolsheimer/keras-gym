import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.value_functions import AtariQ
from keras_gym.policies import EpsilonGreedy  # or BoltzmannPolicy
# from keras_gym.caching import ExperienceReplayBuffer

from dqn_helper import ExperienceArrayBuffer


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=4)


# value function
Q = AtariQ(env, lr=0.00025, gamma=0.99, bootstrap_n=1)
# buffer = ExperienceReplayBuffer(
#     gamma=0.99, bootstrap_n=1, capacity=1000000, batch_size=32, num_frames=4)
buffer = ExperienceArrayBuffer(env, capacity=1000000)
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


def generate_gif(frames, ep):
    """
    Taken from: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756  # noqa: E501
    """
    import os
    import imageio
    from skimage.transform import resize

    for i, frame in enumerate(frames):
        frames[i] = resize(
            frame, (420, 320, 3),
            preserve_range=True, order=0).astype('uint8')
    os.makedirs('data/gifs2', exist_ok=True)
    imageio.mimsave(
        'data/gifs2/ep{:06d}.gif'.format(ep),
        frames,
        duration=1 / 30)


def evaluate(env, ep):
    frames = []
    s = env.reset()
    G = 0.

    for t in range(1000):
        policy.epsilon = 0.01
        a = policy(s)
        s, r, done, info = env.step(a)
        G += r

        frames.append(info['s_orig'][0])

        if done:
            break

    generate_gif(frames, ep)
    print("[EVAL] ep: {}, G: {}, t: {}".format(ep, G, t))


# global step counter
T = 0

for ep in range(num_episodes):
    if ep % 10 == 0 and T > buffer_warmup_period:
        evaluate(env, ep)

    s = env.reset()
    G = 0  # to accumulate return

    for t in range(num_steps):
        policy.epsilon = epsilon(T)
        a = policy(s)
        s_next, r, done, info = env.step(a)

        # counters
        G += r
        T += 1

        buffer.add(info['s_orig'][0], a, r, done, info)
        # buffer.add(s, a, r, done, ep)

        if T > buffer_warmup_period:
            # Q.batch_update(*buffer.sample())
            S, A, R, D, S_next, A_next = buffer.sample()
            Q.batch_update(S, A, R, (1 - D) * 0.99, S_next, A_next)

        if T % target_model_sync_period == 0:
            Q.sync_target_model()

        if done:
            print("ep: {}, T: {}, G: {}, t: {}, epsilon: {}"
                  .format(ep, T, G, t, policy.epsilon))
            break

        s = s_next
