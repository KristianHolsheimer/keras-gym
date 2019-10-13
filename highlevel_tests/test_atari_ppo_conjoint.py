import gym
import keras_gym as km


def test_atari_ppo_conjoint():
    # env with preprocessing
    env = gym.make('PongDeterministic-v4')
    env = km.wrappers.ImagePreprocessor(
        env, height=105, width=80, grayscale=True)
    env = km.wrappers.FrameStacker(env, num_frames=3)
    env = km.wrappers.TrainMonitor(env)

    # show logs from TrainMonitor
    km.enable_logging()

    # actor-critic with shared weights between pi(a|s) and v(s)
    func = km.predefined.AtariFunctionApproximator(env, lr=0.00025)
    actor_critic = km.ConjointActorCritic(
        func,
        gamma=0.99,
        bootstrap_n=10,
        update_strategy='ppo',
        ppo_clip_eps=0.2,
        entropy_beta=0.01)

    v = actor_critic.value_function
    pi = actor_critic.policy

    # we'll use this to temporarily store our experience
    buffer = km.caching.ExperienceReplayBuffer.from_value_function(
        v, capacity=256, batch_size=64)

    # run episodes
    while env.T < 500000:
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

        if env.G > 0:
            break

    assert env.T < 500000, "test_atari_ppo_conjoint didn't converge"
