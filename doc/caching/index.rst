Caching
=======

In RL we often make use of data caching. This might be short-term caching, over
the course of an episode, or it might be long-term caching as is done in
experience replay.


Short-term Caching
------------------

Our short-term caching objects allow us to cache experience within an episode.
For instance :class:`MonteCarloCache <keras_gym.caching.MonteCarloCache>`
caches all transitions collected over an entire episode and then gives us back
the the :math:`\gamma`-discounted returns when the episode
finishes.

Another short-term caching object is :class:`NStepCache
<keras_gym.caching.NStepCache>`, which keeps an :math:`n`-sized sliding window
of transitions that allows us to do :math:`n`-step bootstrapping.


Experience Replay Buffer
------------------------

At the moment, we only have one long-term caching object, which is the
:class:`ExperienceReplayBuffer <keras_gym.caching.ExperienceReplayBuffer>`.
This object can hold an arbitrary number of transitions; the only constraint is
the amount of available memory on your machine.

The way we use learn from the experience stored in the
:class:`ExperienceReplayBuffer <keras_gym.caching.ExperienceReplayBuffer>` is
by sampling from it and then feeding the batch of transitions to our
:term:`function approximator`.



Objects
-------

.. toctree::

    short_term
    experience_replay

