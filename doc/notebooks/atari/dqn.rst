Atari 2600: Pong with DQN
=========================

In this notebook we solve the `PongDeterministic-v4
<https://gym.openai.com/envs/Pong-v0/>`_ environment using deep Q-learning
(`DQN <https://deepmind.com/research/dqn/>`_). We'll use a convolutional neural
net (without pooling) as our function approximator for the :term:`Q-function
<type-II state-action value function>`, see :class:`AtariQ
<keras_gym.value_functions.AtariQ>`.

This notebook periodically generates GIFs, so that we can inspect how the
training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: ../../_static/img/pong.gif
  :alt: Beating Atari 2600 Pong after a few hundred episodes.
  :align: center


To view the notebook in a new tab, click |here|. To interact with the notebook
in Google Colab, hit the "Open in Colab" button below.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/dqn.ipynb
    :alt: Open in Colab

.. raw:: html

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/dqn.html"></iframe>

.. |here| raw:: html

    <a href="../../_static/notebooks/atari/dqn.html" target="_blank" style="font-weight:bold">here</a>

