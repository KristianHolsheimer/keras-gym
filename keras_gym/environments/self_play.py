from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

from ..base.errors import UnavailableActionError, EpisodeDoneError


__all__ = (
    'ConnectFourEnv',
)


class ConnectFourEnv(Env):
    """
    An adversarial environment for playing the `Connect-Four game
    <https://en.wikipedia.org/wiki/Connect_Four>`_.

    Parameters
    ----------
    win_reward : float, optional
        The reward to give out when episode finishes on a win.

    draw_reward : float, optional
        The reward to give out when episode finishes on a draw.

    intermediate_reward : float, optional
        The reward to give out between turns when episode is ongoing.

    Attributes
    ----------
    action_space : gym.spaces.Discrete(7)
        The action space.

    observation_space : MultiDiscrete(nvec)

        The state observation space, representing the position of the current
        player's tokens (``s[1:,:,0]``) and the other player's tokens
        (``s[1:,:,1]``) as well as a mask over the space of actions, indicating
        which actions are available to the current player (``s[0,:,0]``) or the
        other player (``s[0,:,1]``).

        **Note:** The "current" player is relative to whose turn it is, which
        means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap between
        turns.

    max_time_steps : int
        Maximum number of timesteps within each episode.

    available_actions : array of int
        Array of available actions. This list shrinks when columns saturate.

    """  # noqa: E501
    # class attributes
    num_rows = 6
    num_cols = 7
    num_players = 2
    shape = (num_rows + 1, num_cols, num_players)
    action_space = Discrete(num_cols)
    observation_space = MultiDiscrete(nvec=np.full(shape, 2, dtype='uint8'))
    max_time_steps = int(np.ceil(num_rows * num_cols / 2))

    def __init__(
            self,
            win_reward=1.0,
            draw_reward=-0.5,
            intermediate_reward=0.0):

        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.intermediate_reward = intermediate_reward
        self._init_state()

    def reset(self):
        """
        Reset the environment to the starting position.

        Returns
        -------
        s : 3d-array, shape: [num_rows + 1, num_cols, num_players]

            A state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        """
        self._init_state()
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
        s_next : array, shape [6, 7, 2]

            A next-state observation, representing the position of the current
            player's tokens (``s[1:,:,0]``) and the other player's tokens
            (``s[1:,:,1]``) as well as a mask over the space of actions,
            indicating which actions are available to the current player
            (``s[0,:,0]``) or the other player (``s[0,:,1]``).

            **Note:** The "current" player is relative to whose turn it is,
            which means that the entries ``s[:,:,0]`` and ``s[:,:,1]`` swap
            between turns.

        r : float
            Reward associated with the transition
            :math:`(s, a)\\to s_\\text{next}`.

            **Note:** Since "current" player is relative to whose turn it is,
            you need to be careful about aligning the rewards with the correct
            state or state-action pair. In particular, this reward :math:`r` is
            the one associated with the :math:`s` and :math:`a`, i.e. *not*
            aligned with :math:`s_\\text{next}`.

        done : bool
            Whether the episode is done.

        info : dict or None
            A dict with some extra information (or None).

        """
        if self.done:
            raise EpisodeDoneError("please reset env to start new episode")
        if not self.action_space.contains(a):
            raise ValueError("invalid action")
        if a not in self.available_actions:
            raise UnavailableActionError("action is not available")

        # swap players
        self._current_player, self._other_player = (
            self._other_player, self._current_player)

        # update state
        self._state[self._levels[a], a] = self._current_player
        self._prev_action = a

        # run logic
        self.done, reward = self._done_reward(a, self._current_player)
        return self.state, reward, self.done, {'state_id': self.state_id}

    def render(self, *args, **kwargs):
        """
        Render the current state of the environment.

        """
        # lookup for symbols
        symbol = {
            1: u'\u25CF',   # player 1 token (agent)
            2: u'\u25CB',   # player 2 token (adversary)
            -1: u'\u25BD',  # indicator for player 1's last action
            -2: u'\u25BC',  # indicator for player 2's last action
        }

        # render board
        hrule = '+---' * self.num_cols + '+\n'
        board = "  "
        board += "   ".join(
            symbol.get(-(a == self._prev_action) * self._other_player, " ")
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

    @property
    def state(self):
        stacked_layers = np.stack((
            (self._state == self._current_player).astype('uint8'),
            (self._state == self._other_player).astype('uint8'),
        ), axis=-1)  # shape: [num_rows, num_cols, num_players]
        available_actions_mask = np.zeros(
            (1, self.num_cols, self.num_players), dtype='uint8')
        available_actions_mask[0, self.available_actions, :] = 1
        return np.concatenate((available_actions_mask, stacked_layers), axis=0)

    @property
    def state_id(self):
        p = str(self._current_player)
        d = '1' if self.done else '0'
        if self._prev_action is None:
            a = str(self.num_cols)
        else:
            a = str(self._prev_action)
        s = ''.join(self._state.ravel().astype('str'))  # base-3 string
        s = '{:017x}'.format(int(s, 3))  # 17-char hex string
        return p + d + a + s             # 20-char hex string

    def set_state(self, state_id):
        # decode state id
        p = int(state_id[0], 16)
        d = int(state_id[1], 16)
        a = int(state_id[2], 16)
        assert p in (1, 2)
        assert d in (0, 1)
        assert self.action_space.contains(a) or a == self.num_cols
        self._current_player = p    # 1 or 2
        self._other_player = 3 - p  # 2 or 1
        self.done = d == 1
        self._prev_action = None if a == self.num_cols else a
        s = np.base_repr(int(state_id[3:], 16), 3)
        z = np.zeros(self.num_rows * self.num_cols, dtype='uint8')
        z[-len(s):] = np.array(list(s), dtype='uint8')
        self._state = z.reshape((self.num_rows, self.num_cols))
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        for j in range(self.num_cols):
            for i in self._state[::-1, j]:
                if i == 0:
                    break
                self._levels[j] -= 1

    @ property
    def available_actions(self):
        actions = np.argwhere(
            (self._levels >= 0) & (self._levels < self.num_rows)).ravel()
        assert actions.size <= self.num_cols
        return actions

    def _init_state(self):
        self._prev_action = None
        self._current_player = 1
        self._other_player = 2
        self._state = np.zeros((self.num_rows, self.num_cols), dtype='uint8')
        self._levels = np.full(self.num_cols, self.num_rows - 1, dtype='uint8')
        self.done = False

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

        # check vertical: top -> bottom
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            i += 1
            if i > i_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, self.win_reward

        # check horizontal: right -> left
        c = 1
        i, j = i_init, j_init
        for _ in range(3):
            j -= 1
            if j < 0 or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, self.win_reward

        # check horizontal: left -> right
        i, j = i_init, j_init
        for _ in range(3):
            j += 1
            if j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, self.win_reward

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
                return True, self.win_reward

        # check diagonal: NW -> SE
        i, j = i_init, j_init
        for _ in range(3):
            i += 1
            j += 1
            if i > i_max or j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, self.win_reward

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
                return True, self.win_reward

        # check diagonal: SW -> NE
        i, j = i_init, j_init
        for _ in range(3):
            i -= 1
            j += 1
            if i < 0 or j > j_max or self._state[i, j] != player:
                break
            c += 1
            if c == 4:
                return True, self.win_reward

        # check for a draw
        if len(self.available_actions) == 0:
            return True, self.draw_reward

        # this is what's returned throughout the episode
        return False, self.intermediate_reward
