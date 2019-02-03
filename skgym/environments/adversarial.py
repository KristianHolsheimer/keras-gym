from gym.spaces import Discrete, MultiDiscrete
import numpy as np


class NoAdversaryError(Exception):
    pass


class UnavailableActionError(Exception):
    pass


class ConnectFour:
    # class attributes
    num_rows = 6
    num_cols = 7
    shape = (num_rows, num_cols)
    action_space = Discrete(num_cols)
    observation_space = MultiDiscrete(np.full(shape, 3, dtype='int'))

    @ property
    def available_actions(self):
        actions = np.argwhere(self._levels >= 0).ravel()
        assert actions.size <= self.num_cols
        return actions

    def __init__(self, adversary_policy=None, greedy_adversary=True,
                 random_seed=None):
        self.adversary_policy = adversary_policy
        self.greedy_adversary = greedy_adversary

        self.rnd = np.random.RandomState(random_seed)
        self.reset()

    def _adversary_action(self):
        if self.greedy_adversary:
            a = self.adversary_policy.greedy(self.state)
        else:
            a = self.adversary_policy.thompson(self.state)
        if a not in self.available_actions:
            return self._adversary_action()
        self.last_adversary_action = a
        return a

    def reset(self):
        self.state = np.zeros(self.shape, dtype='int')

        # filling levels
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='int')
        self.last_adversary_action = None

        # flip a coin to decide whether adversary starts
        if self.adversary_policy is not None and self.rnd.randint(2):
            a = self._adversary_action()
            self.state[-1, a] = 2
            self._levels[a] -= 1

        return self.state

    def _done_reward(self, a, player):
        assert self.action_space.contains(a)
        assert player in (1, 2)

        # upper bounds
        imax = self.num_rows - 1
        jmax = self.num_cols - 1

        # check for a draw
        if len(self.available_actions) == 0:
            return True, -0.5

        # check vertical
        i, j, c = self._levels[a], a, 1
        for _ in range(3):
            i -= 1
            if i < 0 or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        i = self._levels[a]
        for _ in range(3):
            i += 1
            if i > imax or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        # check horizontal
        i, j, c = self._levels[a], a, 1
        for _ in range(3):
            j -= 1
            if j < 0 or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        j = a
        for _ in range(3):
            j += 1
            if j > jmax or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        # check diagonal: NW/SE
        i, j, c = self._levels[a], a, 1
        for _ in range(3):
            i -= 1
            j -= 1
            if i < 0 or j < 0 or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        i, j = self._levels[a], a
        for _ in range(3):
            i += 1
            j += 1
            if i > imax or j > jmax or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        # check diagonal: NE/SW
        i, j, c = self._levels[a], a, 1
        for _ in range(3):
            i += 1
            j -= 1
            if i > imax or j < 0 or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        i, j = self._levels[a], a
        for _ in range(3):
            i -= 1
            j += 1
            if i < 0 or j > jmax or self.state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, 1.0 if player == 1 else -1.0

        return False, 0.0

    def step(self, action):
        if self.adversary_policy is None:
            raise NoAdversaryError(
                "must specify adversary in order to run the environment")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        if action not in self.available_actions:
            raise UnavailableActionError("action is not available")

        # player's turn
        player = 1
        self.state[self._levels[action], action] = player
        done, reward = self._done_reward(action, player)
        self._levels[action] -= 1
        if done:
            return self.state, reward, done, None

        # adversary's turn
        player = 2
        action = self._adversary_action()
        self.state[self._levels[action], action] = player
        done, reward = self._done_reward(action, player)
        self._levels[action] -= 1

        return self.state, reward, done, None

    def render(self):
        # lookup for symbols
        symbol = {
            1: u'\u25CF',   # player 1 token (agent)
            2: u'\u25CB',   # player 2 token (adversary)
            -1: u'\u25BD',  # indicator for adversary's last action
        }

        # render board
        hrule = '+---' * self.num_cols + '+\n'
        board = "  "
        board += "   ".join(symbol.get(-(a == self.last_adversary_action), " ")
                            for a in range(self.num_cols))
        board += "  \n"
        board += hrule
        for i in range(self.num_rows):
            board += "| "
            board += " | ".join(symbol.get(self.state[i, j], " ")
                                for j in range(self.num_cols))
            board += " |\n"
            board += hrule

        print(board)
