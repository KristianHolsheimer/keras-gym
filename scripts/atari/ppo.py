import gym
import keras_gym as km
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Flatten
from tensorflow.keras import backend as K

# env with preprocessing
env = gym.make('PongDeterministic-v4')
env = km.wrappers.ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = km.wrappers.FrameStacker(env, num_frames=3)
env = km.wrappers.TrainMonitor(env)

# show logs from TrainMonitor
km.enable_logging()


class Func(km.FunctionApproximator):
    def body(self, S):
        def diff_transform(S):
            S = K.cast(S, 'float32') / 255
            M = km.utils.diff_transform_matrix(num_frames=3)
            return K.dot(S, M)

        X = Lambda(diff_transform)(S)
        X = Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')(X)
        X = Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(X)
        X = Flatten()(X)
        X = Dense(units=256, activation='relu')(X)
        return X


# function approximators
func = Func(env, lr=0.00025)
pi = km.SoftmaxPolicy(func, update_strategy='ppo')
v = km.V(func, gamma=0.99, bootstrap_n=10, bootstrap_with_target_model=True)
actor_critic = km.ActorCritic(pi, v)


# we'll use this to temporarily store our experience
buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    value_function=v, capacity=256, batch_size=64)


# run episodes
while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s, use_target_model=True)  # target_model == pi_old
        s_next, r, done, info = env.step(a)

        buffer.add(s, a, r, done, env.ep)

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
    if env.ep % 50 == 0:
        km.utils.generate_gif(
            env=env,
            policy=pi,
            filepath='./data/ppo/gifs/ep{:06d}.gif'.format(env.ep),
            resize_to=(320, 420))
