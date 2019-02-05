n-step Bootstrap with Linear Function Approximator
==================================================

In this notebook we solve the `'CartPole-v0'` environment using the following :math:`n`-step bootstrap algorithms:

- :class:`skgym.algorithms.NStepQLearning`
- :class:`skgym.algorithms.NStepSarsa`
- :class:`skgym.algorithms.NStepExpectedSarsa`

Our Q-function uses scikit-learn's :class:`SGDRegressor <sklearn.linear_model.SGDRegressor>` as its underlying function approximator.


GitHub version
--------------

.. raw:: html

    <p>
    For an up-to-date version of this notebook, see the GitHub version <a href="https://github.com/KristianHolsheimer/scikit-gym/blob/master/notebooks/cartpole-linear-model-nstep-bootstrap.ipynb"  target="_blank" style="font-weight:bold">here</a>.
    </p>


Notebook
--------


.. raw:: html

    <p>
    To view the below notebook in a new tab, click <a href="../_static/notebooks/cartpole-linear-model-nstep-bootstrap.html" target="_blank" style="font-weight:bold">here</a>.
    </p>

    <iframe width="100%" height="600px" src="../_static/notebooks/cartpole-linear-model-nstep-bootstrap.html"></iframe>
