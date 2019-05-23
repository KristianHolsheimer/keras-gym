import sys
import gym
from atari_dqn_standalone_helpers import AtariDQN, Scheduler
# from atari_dqn_standalone_helpers import ExperienceArrayBuffer

from keras_gym.caching import ExperienceReplayBuffer
from keras_gym.preprocessing import ImagePreprocessor, FrameStacker


# environment, experience buffer and value function
env = gym.make('PongDeterministic-v4')
env = ImagePreprocessor(env, height=105, width=80, grayscale=True)
env = FrameStacker(env, num_frames=4)

# buffer = ExperienceArrayBuffer(env, capacity=1000000)
buffer = ExperienceReplayBuffer(capacity=1000000, num_frames=4)

Q = AtariDQN(
    env, learning_rate=0.00025, update_strategy='double_q_learning')

# one object to keep track of all counters
scheduler = Scheduler(
    T_max=3000000,
    experience_replay_warmup_period=50000,
    target_model_sync_period=10000,
    evaluation_period=10000,
)


while True:
    if scheduler.evaluate:
        s_stacked = env.reset()
        s = env.env._s_orig
        assert s_stacked.shape == (105, 80, 4)
        assert s.shape == (210, 160, 3)
        scheduler.reset_G()
        scheduler.clear_frames()
        scheduler.add_frame(s)
        lives = 0
        lost_lives = True

        for t in range(10000):
            a = Q.epsilon_greedy(s, t == 0, epsilon=0.001)
            s_stacked, r, done, info = env.step(a)
            s = info['s_next_orig'][0]
            scheduler.incr_G(r)
            scheduler.add_frame(s)

            if done or t == 9999:
                print("EVAL: t = {}, G = {}, avg(r) = {}"
                      .format(t, scheduler.G, scheduler.G / t))
                sys.stdout.flush()
                break

            lost_lives = info['ale.lives'] < lives
            lives = info['ale.lives']

        scheduler.generate_gif() if t < 9999 else None
        scheduler.clear_frames()

    s_stacked = env.reset()
    s = env.env._s_orig
    scheduler.incr_ep()
    scheduler.reset_G()
    lives = 0
    lost_lives = True

    for t in range(env.spec.max_episode_steps):
        scheduler.incr_T()

        a = Q.epsilon_greedy(s, lost_lives, scheduler.epsilon)
        s_next_stacked, r, done, info = env.step(a)
        s_next = info['s_next_orig'][0]
        scheduler.incr_G(r)

        # buffer.add(s, a, r, done, info)
        buffer.add(s_stacked, a, r, done, scheduler.ep)
        if not scheduler.experience_replay_warmup:
            S, A, Rn, I_next, S_next, A_next = buffer.sample()
            Q.update(S, A, Rn, I_next == 0.0, S_next, A_next)

        if scheduler.sync_target_model:
            Q.sync_target_model(tau=1.0)

        if done:
            break

        # prepare for next timestep
        s = s_next
        s_stacked = s_next_stacked
        lost_lives = info['ale.lives'] < lives
        lives = info['ale.lives']

    scheduler.print()
    if scheduler.done:
        break

env.close()
