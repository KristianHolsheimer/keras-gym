from keras_gym.environments import ConnectFourEnv
from keras_gym.policies import ConnectFourActorCritic
from keras_gym.planning import SearchNode


env = ConnectFourEnv()
ac = ConnectFourActorCritic(env)


# state_id = '20400000000000000099'
# state_id = '2020000d2c2a86ce6400'
# state_id = '10600000000000005609'  # attack
state_id = '20600000000000004d7e'  # defend
# state_id = '106000000001a021e87f'
n = SearchNode(state_id, ac, random_seed=7)


n.env.render()
n.search(n=128)
print('-' * 80)
n.show(depth=3)


s, a, r, done = n.play(tau=0)
n.env.render()
print('-' * 80)
n.show(depth=3)
