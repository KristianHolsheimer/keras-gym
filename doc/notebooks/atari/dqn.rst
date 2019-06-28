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


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/keras-gym/blob/master/notebooks/atari/dqn.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------

.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../../_static/notebooks/atari/dqn.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../../_static/notebooks/atari/dqn.html"></iframe>
