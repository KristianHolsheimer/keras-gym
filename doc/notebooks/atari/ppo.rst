Atari 2600: Pong with PPO
=========================

In this notebook we solve the `PongDeterministic-v4
<https://gym.openai.com/envs/Pong-v0/>`_ environment using a TD actor-critic
algorithm with `PPO <https://openai.com/blog/openai-baselines-ppo/>`_ policy
updates.

We use convolutional neural nets (without pooling) as our function
approximators for the :term:`state value function` :math:`V(s)` and
:term:`policy <updateable policy>` :math:`\pi(a|s)`, see :class:`AtariV
<keras_gym.value_functions.AtariV>` and :class:`AtariPolicy
<keras_gym.policies.AtariPolicy>`. The policy (actor) and value function
(critic) are combined into a single actor-critic object using
:class:`ActorCritic <keras_gym.policies.ActorCritic>`.

This notebook periodically generates GIFs, so that we can inspect how the
training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: ../../_static/img/pong.gif
  :alt: DQN beating Atari 2600 Pong after a few hundred episodes.
  :align: center


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/ppo.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------

.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../../_static/notebooks/atari/ppo.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/ppo.html"></iframe>
