# flake8: noqa
import os
import glob
import numpy as np
import gym
import keras_gym as km
import tensorflow as tf
import scipy.stats as st
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--load_episode', action='store', type=int)
parser.add_argument('-l', '--learning_rate', action='store', type=float, default=1e-5)
args = parser.parse_args()
logger = logging.getLogger('pendulum.ppo')

# abbreviations
keras = tf.keras
K = keras.backend
K.clear_session()


def save_models(models, ep):
    dirpath = 'data/ppo/weights/ep{:08d}'.format(ep)
    os.makedirs(dirpath, exist_ok=True)
    models['train'].save_weights(os.path.join(dirpath, 'train.h5'))
    for a in ('value', 'policy'):
        for b in ('predict', 'target'):
            models[a][b].save_weights(os.path.join(dirpath, f'{a}_{b}.h5'))


def load_models(models, ep=None):
    if ep is None:
        pattern = 'data/ppo/weights/ep*'
        dirpaths = glob.glob(pattern)
        if not dirpaths:
            logger.warn(
                "failed to load model weights from glob pattern '{}'"
                .format(pattern))
            return 0
        dirpath = max(dirpaths)
        ep = int(dirpath.replace('data/ppo/weights/ep', ''))
    else:
        dirpath = 'data/ppo/weights/ep{:08d}'.format(ep)
    logger.info("loading model weights from dir: '{}'".format(dirpath))
    logger.info("will start from episode: {}".format(ep + 1))
    models['train'].load_weights(os.path.join(dirpath, 'train.h5'))
    for a in ('value', 'policy'):
        for b in ('predict', 'target'):
            models[a][b].load_weights(os.path.join(dirpath, f'{a}_{b}.h5'))
    return ep


lr = args.learning_rate
eps = 0.2
gamma = 0.99
tau = 0.1
beta = 0.01
loss_weights = [1, 1]
target_model_sync_period = 16


env = gym.make('Pendulum-v0')
env = km.wrappers.TrainMonitor(env)
km.enable_logging()


# inputs
S = keras.Input(name='S', shape=[3], dtype='float32')
G = keras.Input(name='G', shape=[], dtype='float32')


# computation graph
S2 = keras.layers.Lambda(lambda x: K.pow(x, 2), name='square')(S)
X = keras.layers.Concatenate(name='concat')([S, S2])

X_primary = keras.layers.Dense(name='primary/hidden', units=6, activation='relu')(X)
X_target = keras.layers.Dense(name='target/hidden', units=6, activation='relu')(X)

Z = keras.layers.Dense(name='primary/Z', units=2, kernel_initializer='zeros', activation='tanh')(X_primary)
Z_target = keras.layers.Dense(name='target/Z', units=2, kernel_initializer='zeros', activation='tanh')(X_target)

V = keras.layers.Dense(name='primary/V', units=1, kernel_initializer='zeros')(X_primary)
V_target = keras.layers.Dense(name='target/V', units=1, kernel_initializer='zeros')(X_target)


# custom loss function
def ppo_clip_loss(Adv, Z_target):
    def loss_func(A, Z):
        assert K.ndim(A) == 2
        A = K.squeeze(A, axis=1)
        assert K.ndim(Z) == 2
        assert K.int_shape(Z)[1] == 2
        assert K.ndim(Z_target) == 2
        assert K.int_shape(Z_target)[1] == 2
        mu, logvar = tf.unstack(Z, axis=1)  # logvar == log(sigma^2)
        mu *= 2
        mu_old, logvar_old = tf.unstack(K.stop_gradient(Z_target), axis=1)
        mu_old *= 2
        logpi = -(K.square(A - mu) / K.exp(logvar) + logvar) / 2
        logpi_old = -(K.square(A - mu_old) / K.exp(logvar_old) + logvar_old) / 2
        ratio = K.exp(logpi - logpi_old)
        ratio_clipped = K.clip(ratio, 1 - eps, 1 + eps)
        objective = K.minimum(ratio * Adv, ratio_clipped * Adv)

        # entropy bonus
        entropy = (logvar + K.log(2 * np.pi) + 1) / 2

        return -(objective + beta * entropy)  # flip sign to get a loss

    return loss_func


# models
models = {
    'value': {
        'predict': keras.Model(S, V),
        'target': keras.Model(S, V_target),
    },
    'policy': {
        'predict': keras.Model(S, Z),
        'target': keras.Model(S, Z_target),
    },
    'train': keras.Model([S, G], [Z, V]),
}

Adv = G - V_target
models['train'].compile(
    optimizer=keras.optimizers.Adam(lr),
    # optimizer=keras.optimizers.SGD(lr, momentum=0.9),
    loss=[ppo_clip_loss(Adv, Z_target), keras.losses.MeanSquaredError()],
    loss_weights=loss_weights)


# target model updater
target_weights = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='target')  # list
primary_weights = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='primary')  # list
target_model_sync_op = tf.group(*(
    K.update(wt, wt + tau * (wp - wt))
    for wt, wp in zip(target_weights, primary_weights)))


# load models from earlier run
env.ep = load_models(models, ep=args.load_episode)

# early stopping
successes = 0

for ep in range(env.ep, 100000):

    # run episode
    S = np.array([env.reset()])

    for t in range(env.spec.max_episode_steps):

        # policy
        Z_target = models['policy']['target'].predict_on_batch(S)
        mu, logsigma = Z_target[0]
        mu *= 2
        logsigma *= 2
        try:
            A = st.norm(mu, np.exp(logsigma)).rvs(1)
        except Exception:
            print(mu, logsigma)
            raise

        # one time step
        s_next, r, done, info = env.step(A)
        S_next, R = np.array([s_next]), np.array([r])

        # estimate expected return
        V_next = models['value']['target'].predict_on_batch(S_next)[:, 0]
        G_estimated = R + gamma * V_next  # TD(0) target

        # update
        losses = models['train'].train_on_batch([S, G_estimated], [A, G_estimated])
        losses = dict(zip(models['train'].metrics_names, losses))
        env.record_losses(losses)

        if env.T % target_model_sync_period == 0:
            K.get_session().run(target_model_sync_op)

        if done:
            if env.G > -10:
                successes += 1
                save_models(models, ep)
            else:
                successes = 0
            break

        S = S_next

    if successes >= 10:
        break
