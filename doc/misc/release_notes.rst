Release Notes
=============


v0.2.7
------

This was a *MAJOR* overhaul in which I ported everything from scikit-learn to
Keras. The reason for this is that I was stuck on the implementation of policy
gradient methods due to the lack of flexibility of the scikit-learn ecosystem.
I chose Keras as a replacement, it's nice an modular like scikit-learn,
but in addition it's much more flexible. In particular, the ability to provide
custom loss functions has been the main selling point. Another selling point
was that some environments require more sophisticated neural nets than a
simple MLP, which is readily available in Keras.

- added compatibility wrapper for scikit-learn function approximators
- ported all value functions to use `keras.Model`
- ported predefined models :class:`LinearV <keras_gym.value_functions.LinearV>` and :class:`LinearQ <keras_gym.value_functions.LinearQ>` to keras
- ported algorithms to keras
- ported all notebooks to keras
- changed name of the package `keras-gym` and root module :mod:`keras_gym`

Other changes:

- added propensity score outputs to policy objects
- created a stub for directly updateable policies


v0.2.6
------

- refactored BaseAlgorithm to simplify implementation (at the cost of more code, but it's worth it)
- refactored notebooks: they are now bundled by environment / algo type
- added n-step bootstrap algorithms:

  - :class:`NStepQLearning <keras_gym.algorithms.NStepQLearning>`
  - :class:`NStepSarsa <keras_gym.algorithms.NStepSarsa>`
  - :class:`NStepExpectedSarsa <keras_gym.algorithms.NStepExpectedSarsa>`


v0.2.5
------

- added algorithm: :class:`keras_gym.algorithms.ExpectedSarsa`
- added object: :class:`keras_gym.utils.ExperienceCache`
- rewrote :class:`MonteCarlo <keras_gym.algorithms.MonteCarlo>` to use :class:`ExperienceCache <keras_gym.utils.ExperienceCache>`


v0.2.4
------

- added algorithm: :class:`keras_gym.algorithms.MonteCarlo`


v0.2.3
------

- added algorithm: :class:`keras_gym.algorithms.Sarsa`


v0.2.2
------

- changed doc theme from sklearn to readthedocs


v0.2.1
------

- first working implementation value function + policy + algorithm
- added first working example in a notebook
- added algorithm: :class:`keras_gym.algorithms.QLearning`
