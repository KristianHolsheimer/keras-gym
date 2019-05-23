import gym

from keras_gym.preprocessing import ImagePreprocessor, FrameStacker
from keras_gym.utils import TrainMonitor
from keras_gym.value_functions import AtariQ
from keras_gym.policies import EpsilonGreedy  # or BoltzmannPolicy
from keras_gym.caching import ExperienceReplayBuffer


# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=4)
env = TrainMonitor(env)


# value function
Q = AtariQ(env, lr=0.00025, gamma=0.99, bootstrap_n=1)
buffer = ExperienceReplayBuffer(
    gamma=0.99, bootstrap_n=1, capacity=1000000, batch_size=32, num_frames=4)
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


def generate_gif(env):
    """
    This function was taken from this `blog post <https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756>`_.

    """  # noqa: E501
    import os
    import imageio
    from skimage.transform import resize

    shape = (420, 320, 3)
    duration = 1 / 30

    # collect frames
    frames = []
    s = env.reset()
    for t in range(2000):
        policy.epsilon = 0.01
        a = policy(s)
        s, r, done, info = env.step(a)

        # get original (non-preprocessed) frame, resize and store
        frame = info['s_orig'][0]
        frame = resize(
            frame, shape, preserve_range=True, order=0).astype('uint8')
        frames.append(frame)

        if done:
            break

    os.makedirs('data/gifs', exist_ok=True)
    imageio.mimsave(
        'data/gifs/ep{:06d}.gif'.format(env.ep), frames, duration=duration)


for _ in range(num_episodes):
    if env.ep % 10 == 0 and env.T > buffer_warmup_period:
        generate_gif(env)

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
