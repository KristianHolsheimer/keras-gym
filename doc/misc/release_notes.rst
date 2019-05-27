Release Notes
=============


v0.2.11
-------

- optimized ActorCritic to avoid feeding in :ters:`S` three times instead of once
- removed all mention of ``bootstrap_model``

v0.2.10
-------

This is the second overhaul, a complete rewrite in fact. There was just too
much of the old scikit-gym structure that was standing in the way of progress.

The main thing that changed in this version is that I ditch the notion of an
algorithm. Instead, function approximators carry their own "update strategy".
In the case of Q-functions, this is 'sarsa', 'q_learning' etc., while policies
have the options 'vanilla', 'ppo', etc.

Value functions carry another property that was previously attributed to
algorithm objects. This is the bootstrap-n, i.e. the number of steps over which
to delay bootstrapping.

This new structure accommodates for modularity much much better than the old
structure.

- removed algorithms, replaced by 'bootstrap_n' and 'update_strategy' settings on function approximators
- implemented :class:`ExperienceReplayBuffer <keras_gym.caching.ExperienceReplayBuffer>`
- milestone: added DQN implementation for Atari 2600 envs.
- other than that.. too much to mention. It really was a complete rewrite


v0.2.9
------

- changed definitions of Q-functions to :class:`GenericQ <keras_gym.value_function.GenericQ>` and  :class:`GenericQTypeII <keras_gym.value_function.GenericQTypeII>`
- added option for efficient bootstrapped updating (``bootstrap_model`` argument in value functions, see example usage: :class:`NStepBootstrapV <keras_gym.algorithms.NStepBootstrapV>`)
- renamed :class:`ValuePolicy` to :class:`ValueBasedPolicy <keras_gym.policies.ValueBasedPolicy>`


v0.2.8
------

- implemented base class for updateable policy objects
- implemented first example of updateable policy: :class:`GenericSoftmaxPolicy <keras_gym.policies.GenericSoftmaxPolicy>`
- implemented predefined softmax policy: :class:`LinearSoftmaxPolicy <keras_gym.policies.LinearSoftmaxPolicy>`
- added first policy gradient algorithm: :class:`Reinforce <keras_gym.algorithms.Reinforce>`
- added REINFORCE example notebook
- updated documentation


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
