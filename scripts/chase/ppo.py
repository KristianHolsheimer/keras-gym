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
        s_next, r, done = self.env.step(str(int(action)))
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
actor_critic = km.ConjointActorCritic(
    cnn, update_strategy='ppo', gamma=0.99, bootstrap_n=1)
buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    actor_critic.value_function, capacity=256, batch_size=16)


for ep in range(10000000):
    s = env.reset()

    for t in range(1000):
        a = actor_critic.policy(s, use_target_model=True)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, ep)

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
    if env.ep % 2500 == 0:
        os.makedirs('data/ppo/gifs/', exist_ok=True)
        km.utils.generate_gif(
            env=env,
            policy=actor_critic.policy,
            filepath='data/ppo/gifs/ep{:08d}.gif'.format(env.ep),
            resize_to=(600, 600),
            duration=500)

    # store model weights
    if env.ep % 10000 == 0:
        actor_critic.train_model.save_weights(
            'data/ppo/weights/train_model_{:08d}.h5'.format(env.ep))
        actor_critic.policy.predict_model.save_weights(
            'data/ppo/weights/policy.predict_model_{:08d}.h5'.format(env.ep))
        actor_critic.value_function.predict_model.save_weights(
            'data/ppo/weights/value_function.predict_model_{:08d}.h5'
            .format(env.ep))
