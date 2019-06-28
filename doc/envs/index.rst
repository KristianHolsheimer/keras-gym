Environments
============

This is a collection of environments currently not included in `OpenAI Gym
<https://gym.openai.com/>`_.



Self-Play Environments
----------------------

These environments are typically games. They are implemented in such a way that
can be played from a single-player perspective. The environment switches the
*current player* and *opponent* between turns. The way to picture this is that
the environment swaps color of all pieces between turns, so that the agent
always gets the perspective of the player whose turn it is. The first such
environment we include is the :class:`ConnectFourEnv
<keras_gym.envs.ConnectFourEnv>`.


Objects
-------

.. toctree::

    self_play
