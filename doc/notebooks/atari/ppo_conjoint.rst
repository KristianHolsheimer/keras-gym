Atari 2600: Pong with PPO (with shared weights)
===============================================

In this notebook we solve the `PongDeterministic-v4
<https://gym.openai.com/envs/Pong-v0/>`_ environment using a TD actor-critic
algorithm with `PPO <https://openai.com/blog/openai-baselines-ppo/>`_ policy
updates.

We use convolutional neural nets (without pooling) as our function
approximators, see :class:`AtariFunctionApproximator
<keras_gym.value_functions.AtariFunctionApproximator>`. This notebook is almost
identical to :doc:`this one <ppo>`, except that here we use a conjoint
actor-critic :class:`ConjointActorCritic <keras_gym.ConjointActorCritic>`
rather instead of a generic one :class:`ActorCritic <keras_gym.ActorCritic>`.
This means that the policy and value function share the bulk of the neural net.
They only differ in the :term:`heads <head>`, i.e. their final dense layer.

This notebook periodically generates GIFs, so that we can inspect how the
training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: ../../_static/img/pong.gif
  :alt: Beating Atari 2600 Pong after a few hundred episodes.
  :align: center


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/ppo_conjoint.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------

.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../../_static/notebooks/atari/ppo_conjoint.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/ppo_conjoint.html"></iframe>
