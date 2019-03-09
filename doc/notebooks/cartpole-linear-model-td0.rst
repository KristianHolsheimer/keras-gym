Cartpole TD(0) with Linear Function Approximator
================================================

In this notebook we solve the `CartPole-v0 <https://gym.openai.com/envs/CartPole-v0/>`_ environment using the following TD(0) algorithms:

- :class:`keras_gym.algorithms.QLearning`
- :class:`keras_gym.algorithms.Sarsa`
- :class:`keras_gym.algorithms.ExpectedSarsa`

Our Q-function uses a Keras implementation of a linear regression model as its underlying function approximator.

.. note::

    One of the function approximators in this notebook is a *type-II* model instead of the default type-I, which models the Q-function as a mapping :math:`s\mapsto Q(s,.)` rather than :math:`(s, a)\mapsto Q(s, a)`.

    I didn't have a lot of success getting this to converge properly. In other words, please don't expect consistent performance on CartPole with this specific function approximator. Having said that, I do expect type-II models to be able to outperform type-I ones on some other environments (perhaps environments with a larger action space).


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/keras-gym/blob/master/notebooks/cartpole-linear-model-td0.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------


.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../_static/notebooks/cartpole-linear-model-td0.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../_static/notebooks/cartpole-linear-model-td0.html"></iframe>
