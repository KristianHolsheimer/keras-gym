import sys
import gym
from atari_dqn_standalone_helpers import (
    ExperienceArrayBuffer, AtariDQN, Scheduler)


# environment, experience buffer and value function
env = gym.make('PongDeterministic-v4')
# env = gym.make('BreakoutDeterministic-v4')
# env = gym.make('MsPacman-v0')
buffer = ExperienceArrayBuffer(env, capacity=1000000)
Q = AtariDQN(env, learning_rate=0.00025, update_strategy='q_learning')

# one object to keep track of all counters
scheduler = Scheduler(
    T_max=3000000,
    experience_replay_warmup_period=50000,
    target_model_sync_period=10000,
    evaluation_period=10000,
)


while True:
    if scheduler.evaluate:
        s = env.reset()
        scheduler.reset_G()
        scheduler.clear_frames()
        scheduler.add_frame(s)

        for t in range(10000):
            a = Q.epsilon_greedy(s, t == 0, epsilon=0.001)
            s, r, done, info = env.step(a)
            scheduler.incr_G(r)
            scheduler.add_frame(s)

            if done or t == 9999:
                print("EVAL: t = {}, G = {}, avg(r) = {}"
                      .format(t, scheduler.G, scheduler.G / t))
                sys.stdout.flush()
                break

        scheduler.generate_gif() if t < 9999 else None
        scheduler.clear_frames()

    s = env.reset()
    scheduler.incr_ep()
    scheduler.reset_G()

    for t in range(env.spec.max_episode_steps):
        scheduler.incr_T()

        a = Q.epsilon_greedy(s, t == 0, scheduler.epsilon)
        s_next, r, done, info = env.step(a)
        scheduler.incr_G(r)

        buffer.add(s, a, r, done, info)
        if not scheduler.experience_replay_warmup:
            Q.update(*buffer.sample(n=32))

        if scheduler.sync_target_model:
            Q.sync_target_model(tau=1.0)

        if done:
            break

        # prepare for next timestep
        s = s_next

    scheduler.print()
    if scheduler.done:
        break

env.close()
