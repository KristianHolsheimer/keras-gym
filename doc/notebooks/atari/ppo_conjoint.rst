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


To view the notebook in a new tab, click |here|. To interact with the notebook
in Google Colab, hit the "Open in Colab" button below.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/ppo_conjoint.ipynb
    :alt: Open in Colab

.. raw:: html

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/ppo_conjoint.html"></iframe>

.. |here| raw:: html

    <a href="../../_static/notebooks/atari/ppo_conjoint.html" target="_blank" style="font-weight:bold">here</a>
