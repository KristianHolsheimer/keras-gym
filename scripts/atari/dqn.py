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


func = Func(env, lr=0.00025)

q = km.QTypeII(
    function_approximator=func,
    gamma=0.99,
    bootstrap_n=1,
    bootstrap_with_target_model=True)

buffer = km.caching.ExperienceReplayBuffer.from_value_function(
    value_function=q,
    capacity=1000000,
    batch_size=32)

policy = km.EpsilonGreedy(q)


# DQN update schedule
buffer_warmup_period = 50000
target_model_sync_period = 10000


# DQN exploration schedule
def epsilon(T):
    """ stepwise linear annealing """
    M = 1000000
    if T < M:
        return 1 - 0.9 * T / M
    if T < 2 * M:
        return 0.1 - 0.09 * (T - M) / M
    return 0.01


while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
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
    if env.ep % 10 == 0 and env.T > buffer_warmup_period:
        km.utils.generate_gif(
            env=env,
            policy=policy.set_epsilon(0.01),
            filepath='./data/dqn/gifs/ep{:06d}.gif'.format(env.ep),
            resize_to=(320, 420))
