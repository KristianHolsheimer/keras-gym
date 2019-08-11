import os

import gym_chase  # noqa: F401
import keras_gym as km
import gym
import numpy as np
import tensorflow as tf
from keras_gym.base.mixins import AddOrigStateToInfoDictMixin
from matplotlib import pyplot as plt

keras = tf.keras
K = keras.backend
K.clear_session()


class ChasePreprocessor(gym.Wrapper, AddOrigStateToInfoDictMixin):
    NUM_LAYERS = 4
    WIDTH = 20
    HEIGHT = 20
    SHAPE = (HEIGHT, WIDTH, NUM_LAYERS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.MultiDiscrete(
            nvec=np.full(self.SHAPE, 2, dtype='int'))
        self._seed = 0

    def _preprocess(self, s):
        assert s.shape == (self.HEIGHT, self.WIDTH)
        s_preprocessed = np.zeros(self.SHAPE)
        for i in range(self.NUM_LAYERS):
            s_preprocessed[:, :, i] = s == i + 1
        return s_preprocessed

    def _image(self, s):
        r, y, b = np.array([[255, 0, 0], [255, 215, 0], [0, 0, 255]])

        img = np.zeros((self.HEIGHT, self.WIDTH, 3))
        img[:, :] += np.einsum('k,ij->ijk', y, s[:, :, 0] + s[:, :, 1])
        img[:, :] += np.einsum('k,ij->ijk', r, s[:, :, 2])
        img[:, :] += np.einsum('k,ij->ijk', b, s[:, :, 3])
        return img.astype('uint8')

    def reset(self):
        self._seed += 1
        s = self.env.reset(self._seed)
        s = self._preprocess(s)
        self._s_orig = self._image(s)
        return s

    def step(self, action):
        s_next, r, done = self.env.step(int(action))
        info = {}
        s_next = self._preprocess(s_next)
        self._s_next_orig = self._image(s_next)
        self._add_orig_to_info_dict(info)
        return s_next, r, done, info

    def render(self, *args, **kwargs):
        img = self._image(self._preprocess(self.unwrapped.arena))
        plt.imshow(img)
        plt.grid(None)
        plt.axis('off')
        plt.show()


class CNN(km.FunctionApproximator):
    def body(self, S, variable_scope):
        assert variable_scope in ('primary', 'target')

        def v(name):
            return '{}/{}'.format(variable_scope, name)

        layers = [
            keras.layers.Lambda(lambda X: K.cast(X, 'float32')),
            keras.layers.Conv2D(
                name=v('conv1'), filters=16, kernel_size=6, strides=3,
                activation='relu'),
            keras.layers.Conv2D(
                name=v('conv2'), filters=32, kernel_size=4, strides=2,
                activation='relu'),
            keras.layers.Flatten(name=v('flatten')),
            keras.layers.Dense(
                name=v('dense1'), units=256, activation='relu')]

        # forward pass
        X = S
        for layer in layers:
            X = layer(X)

        return X


# environment [https://github.com/axb2035/gym-chase]
env = gym.make('Chase-v0')
env = ChasePreprocessor(env)
env = km.wrappers.TrainMonitor(env)

# show logs from TrainMonitor
km.enable_logging()


# function approximator
cnn = CNN(env, lr=0.00025)

q = km.QTypeII(
    function_approximator=cnn,
    gamma=0.99,
    bootstrap_n=1,
    bootstrap_with_target_model=True,
    update_strategy='q_learning')

buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    q, capacity=1000000, batch_size=32)

policy = km.EpsilonGreedy(q)


# exploration schedule
def epsilon(T):
    """ stepwise linear annealing """
    M = 1000000
    if T < M:
        return 1 - 0.9 * T / M
    if T < 2 * M:
        return 0.1 - 0.09 * (T - M) / M
    return 0.01


# static parameters
num_episodes = 10000000
num_steps = env.spec.max_episode_steps or 1000
buffer_warmup_period = 50000
target_model_sync_period = 10000


for _ in range(num_episodes):
    s = env.reset()

    for t in range(num_steps):
        policy.epsilon = epsilon(env.T)
        a = policy(s)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)

        if env.T > buffer_warmup_period:
            q.batch_update(*buffer.sample())

        if env.T % target_model_sync_period == 0:
            q.sync_target_model()

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.ep % 2500 == 0:
        os.makedirs('data/dqn/gifs/', exist_ok=True)
        km.utils.generate_gif(
            env=env,
            policy=policy.greedy,
            filepath='data/dqn/gifs/ep{:08d}.gif'.format(env.ep),
            resize_to=(600, 600),
            duration=500)

    # store model weights
    if env.ep % 10000 == 0:
        os.makedirs('data/dqn/weights/', exist_ok=True)
        q.train_model.save_weights(
            'data/dqn/weights/train_model_{:08d}.h5'.format(env.ep))
        q.predict_model.save_weights(
            'data/dqn/weights/predict_model_{:08d}.h5'.format(env.ep))
