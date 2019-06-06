from gym.spaces import Discrete, MultiDiscrete, Tuple, MultiBinary
import numpy as np

from ..base.errors import (
    MissingAdversaryError, UnavailableActionError, EpisodeDoneError)


__all__ = (
    'ConnectFourEnv',
)


class ConnectFourEnv:
    """
    An adversarial environment for playing the `Connect-Four game
    <https://en.wikipedia.org/wiki/Connect_Four>`_.

    Parameters
    ----------
    adversary_policy : Policy object
        The adversary that we play against.

    greedy_adversary : bool, optional
        Whether to select adversarial actions greedily. If `False`, the
        adversary will select actions using thompson sampling.

    win_reward : float, optional
        The reward to give out when episode finishes on a win.

    loss_reward : float, optional
        The reward to give out when episode finishes on a loss.

    draw_reward : float, optional
        The reward to give out when episode finishes on a draw.

    intermediate_reward : float, optional
        The reward to give out between turns when episode is ongoing.

    Attributes
    ----------
    action_space : gym.spaces.Discrete(7)
        The action space.

    observation_space : Tuple((MultiDiscrete([[2, 2, 2, 2, 2, 2, 2], ...]), MultiBinary(7)))
        The state observation space, representing the position of player 1
        tokens (``s[0][:,:,0]``) and player 2 tokens (``s[0][:,:,1]``) as well
        as a mask over the space of actions, indicating which actions are
        available (``s[1]``).

    max_time_steps : int
        Maximum number of timesteps within each episode.

    available_actions : array of int
        Array of available actions. This list shrinks when columns saturate.

    """  # noqa: E501
    # class attributes
    num_rows = 6
    num_cols = 7
    num_layers = 2
    shape = (num_rows, num_cols, num_layers)
    action_space = Discrete(num_cols)
    observation_space = Tuple((
        MultiDiscrete(np.full(shape, 2, dtype='int')),  # board configurations
        MultiBinary(num_cols),                          # available actions
    ))

    max_time_steps = int(np.ceil(num_rows * num_cols / 2))

    def __init__(self, adversary_policy=None, greedy_adversary=True,
                 win_reward=1.0, loss_reward=-1.0, draw_reward=-0.5,
                 intermediate_reward=0.0, random_seed=None):

        self.adversary_policy = adversary_policy
        self.greedy_adversary = greedy_adversary
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.intermediate_reward = intermediate_reward
        self.rnd = np.random.RandomState(random_seed)
        self._init_state()

    @ property
    def available_actions(self):
        actions = np.argwhere(self._levels >= 0).ravel()
        assert actions.size <= self.num_cols
        return actions

    @property
    def state(self):
        stacked_layers = np.stack((
            (self._state == 1).astype('uint8'),  # player 1 layer
            (self._state == 2).astype('uint8'),  # player 2 layer
        ), axis=-1)
        available_actions_mask = np.zeros(self.num_cols, dtype='uint8')
        available_actions_mask[self.available_actions] = 1
        return stacked_layers, available_actions_mask

    def _init_state(self):
        self._state = np.zeros((self.num_rows, self.num_cols), dtype='int')
        self.last_adversary_action = None
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='int')
        self._done = False

    def reset(self):
        """
        Reset the environment to the starting position.

        Returns
        -------
        s : array, shape [6, 7]
            The state representation. Each cell can take either one of three
            values: 0, 1, 2. An empty cell is indicated by 0, a cell filled by
            the agent is set to 1 and a cell filled by the adversary is set to
            2.

        """
        self._init_state()

        # flip a coin to decide whether adversary starts
        if self.adversary_policy is not None and self.rnd.randint(2):
            a = self._adversary_action()
            self._state[-1, a] = 2
            self._levels[a] -= 1

        return self.state

    def step(self, a):
        """
        Take one step in the MDP, following the single-player convention from
        gym.

        Parameters
        ----------
        a : int, options: {0, 1, 2, 3, 4, 5, 6}
            The action to be taken. The action is the zero-based count of the
            possible insertion slots, starting from the left of the board.

        Returns
        -------
        s_next : array, shape [6, 7]
            The state representation. Each cell can take either one of three
            values: 0, 1, 2. An empty cell is indicated by 0, a cell filled by
            the agent is set to 1 and a cell filled by the adversary is set to
            2.

        r : float
            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

        done : bool
            Whether the episode is done.

        info : dict or None
            A dict with some extra information (or None).

        """
        if self._done:
            raise EpisodeDoneError("please reset env to start new episode")
        if self.adversary_policy is None:
            raise MissingAdversaryError(
                "must specify adversary in order to run the environment")
        if not self.action_space.contains(a):
            raise ValueError("invalid action")
        if a not in self.available_actions:
            raise UnavailableActionError("action is not available")

        # player's turn
        player = 1
        self._state[self._levels[a], a] = player
        self._done, reward = self._done_reward(a, player)
        if self._done:
            return self.state, reward, self._done, {}

        # adversary's turn
        player = 2
        a = self._adversary_action()
        self._state[self._levels[a], a] = player
        self._done, reward = self._done_reward(a, player)

        return self.state, reward, self._done, {}

    def render(self):
        """ Render the current state of the environment. """

        # lookup for symbols
        symbol = {
            1: u'\u25CF',   # player 1 token (agent)
            2: u'\u25CB',   # player 2 token (adversary)
            -1: u'\u25BD',  # indicator for adversary's last action
        }

        # render board
        hrule = '+---' * self.num_cols + '+\n'
        board = "  "
        board += "   ".join(
            symbol.get(-int(a == self.last_adversary_action), " ")
            for a in range(self.num_cols))
        board += "  \n"
        board += hrule
        for i in range(self.num_rows):
            board += "| "
            board += " | ".join(
                symbol.get(self._state[i, j], " ")
                for j in range(self.num_cols))
            board += " |\n"
            board += hrule
        board += "  0   1   2   3   4   5   6  \n"  # actions

        print(board)

    def _adversary_action(self, last_attempt=None):
        # create the state as seen from the adversary (swap player layers)
        stacked_layers, available_actions_mask = self.state
        s_adversary = stacked_layers[:, :, ::-1], available_actions_mask

        if self.greedy_adversary:
            a = self.adversary_policy.greedy(s_adversary)
        else:
            a = self.adversary_policy(s_adversary)

        if a not in self.available_actions:
            a = self.rnd.choice(self.available_actions)

        self.last_adversary_action = a
        return a

    def _done_reward(self, a, player):
        """
        Check whether the last action `a` by player `player` resulted in a win,
        loss or draw for player 1 (the agent). This contains the main logic and
        implements the rules of the game.

        """
        assert self.action_space.contains(a)
        assert player in (1, 2)

        # init values and upper bounds
        i_init = self._levels[a]
        j_init = a
        i_max = self.num_rows - 1
        j_max = self.num_cols - 1

        # update filling levels
        self._levels[a] -= 1

        # win/lose?
        reward = self.win_reward if player == 1 else self.loss_reward

        # check vertical: top -> bottom
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            i += 1
            if i > i_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check horizontal: right -> left
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            j -= 1
            if j < 0 or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check horizontal: left -> right
        i, j = i_init, j_init
        for _ in range(3):
            j += 1
            if j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check diagonal: SE -> NW
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            i -= 1
            j -= 1
            if i < 0 or j < 0 or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check diagonal: NW -> SE
        i, j = i_init, j_init
        for _ in range(3):
            i += 1
            j += 1
            if i > i_max or j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check diagonal: NE -> SW
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            i += 1
            j -= 1
            if i > i_max or j < 0 or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check diagonal: SW -> NE
        i, j = i_init, j_init
        for _ in range(3):
            i -= 1
            j += 1
            if i < 0 or j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, reward

        # check for a draw
        if len(self.available_actions) == 0:
            return True, self.draw_reward

        # this is what's returned throughout the episode
        return False, self.intermediate_reward
