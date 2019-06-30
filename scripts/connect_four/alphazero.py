import keras_gym as km


env = km.envs.ConnectFourEnv()
func = km.predefined.ConnectFourFunctionApproximator(env, lr=0.001)
ac = km.ConjointActorCritic(func)


# state_id = '20400000000000000099'
# state_id = '2020000d2c2a86ce6400'
# state_id = '10600000000000005609'  # attack
state_id = '20600000000000004d7e'  # defend
# state_id = '106000000001a021e87f'
n = km.planning.MCTSNode(state_id, ac, random_seed=7)


n.env.render()
# n.search(n=64)
# n.show(max_depth=2)
# n.play(tau=0)
# print('-' * 80)
# n.show(max_depth=2)
# print('-' * 80)
# # n.search(n=64)
# for _ in range(10):
#     n.search(1)
# print('-' * 80)
# n.show(max_depth=2)


for t in range(100):
    n.search(n=100)
    # n.show(2)
    s, a, pi, r, done = n.play(tau=0)
    n.env.render()

    if done:
        break

print(t, r, done)
