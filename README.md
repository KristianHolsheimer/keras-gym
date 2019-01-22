# scikit-gym
*Plug-n-Play Reinforcement Learning in Python*


## Documentation

For the full documentation, go to [scikit-gym.readthedocs.io](https://scikit-gym.readthedocs.io/)


## Install

Install using pip:

```
$ pip install -U scikit-gym
```
or install from a fresh clone
```
$ git clone https://github.com/KristianHolsheimer/scikit-gym.git
$ pip install -e ./scikit-gym
```

## Examples

Check out the [notebooks](notebooks/) for examples. These are also included in
the documentation:

* https://scikit-gym.readthedocs.io/notebooks/


## TODO

* add support for continuous action spaces
* check whether the above example still works
* ~implement experience cache for MC implement experience-replay type algorithms~
* implement sparse one-hot vectors
* implement `_to_vec` for `gym.spaces.Dict` space.
* fix slow monte-carlo algorithm (standalone benchmark in notebook is faster)
