.. automodule:: keras_gym.environments


Environments
============

This is a collection of environments currently not included in `OpenAI Gym
<https://gym.openai.com/>`_.



Self-Play Environments
----------------------

These environments are typically games. They are implemented in such a way that
can be played from a single-player perspective. The environment switches the
*current player* and *opponent* between turns. The way to picture this is that
the environment keeps rotating the game board 180 degrees between turns, so
that the agents always get the perspective of the player whose turn it is. The
first such environment we include is the :class:`ConnectFourEnv
<keras_gym.environments.ConnectFourEnv>`.


References
----------

.. toctree::
    :maxdepth: 2
    :glob:

    self_play
