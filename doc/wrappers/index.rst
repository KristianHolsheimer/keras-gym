Wrappers
========

OpenAI gym provides a nice modular interface to extend existing environments
using `environment wrappers
<https://github.com/openai/gym/tree/master/gym/wrappers>`_. Here we list some
wrappers that are used throughout the **keras-gym** package.


Preprocessors
-------------

The default preprocessor tries to create a feature vector from any environment
state observation *on a best-effort basis*. For instance, if the observation
space is discrete :math:`s\in\{0, 1, \dots, n-1\}`, it will create a one-hot
encoded vector such that the wrapped environment yields state observations
:math:`s\in\mathbb{R}^n`.

.. code::

    import gym
    import keras_gym as km

    env = gym.make('FrozenLake-v0')
    env = km.wrappers.DefaultPreprocessor(env)

    s = env.unwrapped.reset()  # s == 0
    s = env.reset()            # s == [1, 0, 0, ..., 0]


Other preprocessors that are particularly useful when dealing with video input
are :class:`ImagePreprocessor <keras_gym.wrappers.ImagePreprocessor>` and
:class:`FrameStacker <keras_gym.wrappers.FrameStacker>`. For instance, for
Atari 2600 environments we usually apply preprocessing as follows:

.. code::

    env = gym.make('PongDeterministic-v4')
    env = km.wrappers.ImagePreprocessor(env, height=105, width=80, grayscale=True)
    env = km.wrappers.FrameStacker(env, num_frames=4)

    s = env.unwrapped.reset()  # s.shape == (210, 160, 3)
    s = env.reset()            # s.shape == (105,  80, 4)


The first wrapper does down-scaling and grayscaling on each input frame. The
second wrapper then stacks consecutive frames together, which allows for the
function approximator to learn velocities/accelerations as well as positions
for each input pixel.


Monitors
--------

Another type of environment wrapper is a monitor, which is used to keep track
of the progress of the training process. At the moment, **keras-gym** only
provides a generic train monitor called :class:`TrainMonitor
<keras_gym.wrappers.TrainMonitor>`



Objects
-------

.. toctree::

    preprocessors
    monitors
