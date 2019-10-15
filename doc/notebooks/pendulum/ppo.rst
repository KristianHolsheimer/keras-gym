Pendulum with PPO
=================

In this notebook we solve the `Pendulum-v0
<https://gym.openai.com/envs/Pendulum-v0/>`_ environment using a TD actor-critic
algorithm with `PPO <https://openai.com/blog/openai-baselines-ppo/>`_ policy
updates.

We use a simple multi-layer percentron as our function
approximators for the :term:`state value function` :math:`v(s)` and
:term:`policy <updateable policy>` :math:`\pi(a|s)` implemented by :class:`GaussianPolicy <keras_gym.GaussianPolicy>`.

This algorithm is slow to converge (if it does at all). You should start to see
improvement in the average return after about 150k timesteps. Below you'll see
a particularly succesful episode:

.. image:: ../../_static/img/pendulum.gif
  :alt: A particularly succesful episode of Pendulum.
  :align: center


To view the notebook in a new tab, click |here|. To interact with the notebook
in Google Colab, hit the "Open in Colab" button below.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/KristianHolsheimer/keras-gym/blob/master/notebooks/pendulum/ppo.ipynb
    :alt: Open in Colab

.. raw:: html

    <iframe width="100%" height="600px" src="../../_static/notebooks/pendulum/ppo.html"></iframe>

.. |here| raw:: html

    <a href="../../_static/notebooks/pendulum/ppo.html" target="_blank" style="font-weight:bold">here</a>
