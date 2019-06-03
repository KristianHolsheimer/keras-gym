Atari 2600: Pong with PPO (with shared weights)
===============================================

In this notebook we solve the `PongDeterministic-v4
<https://gym.openai.com/envs/Pong-v0/>`_ environment using a TD actor-critic
algorithm with `PPO <https://openai.com/blog/openai-baselines-ppo/>`_ policy
updates.

We use a **joint** convolutional neural net (without pooling) as our function
approximator for the :term:`state value function` :math:`V(s)` and
:term:`policy <updateable policy>` :math:`\pi(a|s)`. Instead of defined the
policy and value function separately, we use a single actor-critic object that
shares all but the last layer of weights between the actor and critic, see
:class:`AtariActorCritic <keras_gym.policies.AtariActorCritic>`.

Although this version doesn't significantly outperform the naive actor-critic
(without shared weights), we include it here to illustrate it as a common use
case.

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
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/ppo_shared_weights.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------

.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../../_static/notebooks/atari/ppo_shared_weights.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/ppo_shared_weights.html"></iframe>
