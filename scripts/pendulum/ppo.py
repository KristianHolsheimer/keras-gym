import argparse
import gym
import keras_gym as km
import tensorflow as tf

# common abbreviations
keras = tf.keras
K = keras.backend


###############################################################################
# input args
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--entropy_beta', type=float, default=0.01)
parser.add_argument('--bootstrap_n', type=int, default=5)
parser.add_argument('--policy_loss_weight', type=float, default=1.)
parser.add_argument('--value_loss_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--minibatch_size', type=int, default=16)
parser.add_argument('--target_sync_tau', type=float, default=0.1)
parser.add_argument('--id', type=int)


args = parser.parse_args()

tensorboard_dir = (
    '/tmp/tensorboard/' + (f'ID={args.id};' if args.id is not None else '') +
    f'LR={args.learning_rate};'
    f'N={args.bootstrap_n};'
    f'GAMMA={args.gamma};'
    f'TAU={args.target_sync_tau};'
    f'BATCH={args.batch_size};'
    f'MBATCH={args.minibatch_size};'
    f'ENTBETA={args.entropy_beta};'
)

env = gym.make('Pendulum-v0')
env = km.wrappers.BoxActionsToReals(env)
env = km.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir)
km.enable_logging()


###############################################################################
# function approximator
###############################################################################

class MLP(km.FunctionApproximator):
    def body(self, X):
        X = keras.layers.Lambda(
            lambda x: K.concatenate([x, K.square(x)], axis=1))(X)
        X = keras.layers.Dense(units=6, activation='tanh')(X)
        X = keras.layers.Dense(units=6, activation='tanh')(X)
        return X


mlp = MLP(env, lr=args.learning_rate)
pi = km.GaussianPolicy(mlp, update_strategy='ppo')
v = km.V(mlp, gamma=args.gamma, bootstrap_n=args.bootstrap_n)
ac = km.ActorCritic(pi, v)


buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    value_function=v, capacity=args.batch_size, batch_size=args.minibatch_size)


###############################################################################
# run
###############################################################################

while env.T < 1000000:
    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=True)
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)
        if len(buffer) >= buffer.capacity:
            # use 4 epochs per round
            num_batches = int(4 * buffer.capacity / buffer.batch_size)
            for _ in range(num_batches):
                ac.batch_update(*buffer.sample())
            buffer.clear()
            pi.sync_target_model(tau=args.target_sync_tau)

        if done:
            break

        s = s_next
