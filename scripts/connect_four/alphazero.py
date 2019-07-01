import keras_gym as km
import numpy as np


env = km.envs.ConnectFourEnv()
func = km.predefined.ConnectFourFunctionApproximator(env, lr=0.001)
ac = km.ConjointActorCritic(func)
cache = km.caching.MonteCarloCache(env, gamma=1)


# state_id = '20400000000000000099'
# state_id = '2020000d2c2a86ce6400'
# state_id = '10600000000000005609'  # attack
# state_id = '20600000000000004d7e'  # defend
# state_id = '106000000001a021e87f'
# n = km.planning.MCTSNode(state_id, ac, random_seed=7)
n = km.planning.MCTSNode(ac, random_seed=17, c_puct=3.5)

n.env.render()


for _ in range(1000):
    n.reset()
    for t in range(env.max_time_steps):
        n.search(n=56)
        n.show(2)
        s, pi, r, done = n.play(tau=0)
        cache.add(s, pi, r, done)
        n.env.render()

        if done:
            G = np.array([r])
            In, S_next = np.array([0]), np.array([s])  # dummy values
            while cache:
                S, P, _ = cache.pop()
                ac.batch_update(S, P, G, In, S_next)
                G = -G  # flip sign for opponent

            print("*** GAME OVER ***")
            break
