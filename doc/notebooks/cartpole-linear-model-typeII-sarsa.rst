Cartpole Sarsa with Linear Model (Type II)
==========================================

In this notebook we solve the `'CartPole-v0'` environment using the
:class:`Sarsa <skgym.algorithms.Sarsa>` algorithm, where our Q-function uses
scikit-learn's :class:`SGDRegressor <sklearn.linear_model.SGDRegressor>` as its
underlying function approximator. What's more, this notebook trains a *type-II*
model instead of the default type-I. This means that it models the Q-function
as a mapping :math:`s\\mapsto Q(s,.)` rather than
:math:`(s, a)\\mapsto Q(s, a)`.


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/scikit-gym/blob/master/notebooks/cartpole-linear-model-typeII-sarsa.ipynb" target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------


.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../_static/notebooks/cartpole-linear-model-typeII-sarsa.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../_static/notebooks/cartpole-linear-model-typeII-sarsa.html"></iframe>
