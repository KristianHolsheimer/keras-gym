.. automodule:: keras_gym.environments


Environments
============

This is a collection of environments currently not included in `OpenAI Gym
<https://gym.openai.com/>`_. The environments included in here are typically
those that depend on the specific structure of the keras-gym package.



Adversarial Environments
------------------------

These environments depend on an adversarial policy in order to be able to run
it. This goes beyond what Gym can support without the incorporation of policy
objects. The first such environment we include is the :class:`ConnectFour
<keras_gym.environments.ConnectFour>`.


References
----------

.. toctree::
    :maxdepth: 2
    :glob:

    adversarial
