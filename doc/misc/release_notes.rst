Release Notes
=============


v0.2.6
------

- refactored BaseAlgorithm to simplify implementation (at the cost of more code, but it's worth it)
- refactored notebooks: they are now bundled by environment / algo type
- added n-step bootstrap algorithms:

  - :class:`NStepQLearning <skgym.algorithms.NStepQLearning>`
  - :class:`NStepSarsa <skgym.algorithms.NStepSarsa>`
  - :class:`NStepExpectedSarsa <skgym.algorithms.NStepExpectedSarsa>`


v0.2.5
------

- added algorithm: :class:`skgym.algorithms.ExpectedSarsa`
- added object: :class:`skgym.utils.ExperienceCache`
- rewrote :class:`MonteCarlo <skgym.algorithms.MonteCarlo>` to use :class:`ExperienceCache <skgym.utils.ExperienceCache>`


v0.2.4
------

- added algorithm: :class:`skgym.algorithms.MonteCarlo`


v0.2.3
------

- added algorithm: :class:`skgym.algorithms.Sarsa`


v0.2.2
------

- changed doc theme from sklearn to readthedocs


v0.2.1
------

- first working implementation value function + policy + algorithm
- added first working example in a notebook
- added algorithm: :class:`skgym.algorithms.QLearning`
