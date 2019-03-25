.. automodule:: keras_gym.algorithms


Algorithms
==========

The main goal of this package is minimize the amount of boiler-plate code you
need to write in order to try out an RL algorithm. One way of accomplishing
this goal is to outsource function approximation as much as possible. For this,
we chose to use `Keras <https://keras.io/>`_, because of its excellent balance
of flexibility and modularity.

The way to add your own algorithm should be easy, but you of course you can
also grab one off the shelve. To look at the algorithms that have been
implemented so far, either to use directly or to use as a template for your own
algorithm, click the **source code** button on the class definitions listed in
the `References`_ below.

For an exhaustive overview of RL algorithms, check out the seminal `Sutton &
Barto RL book <http://incompleteideas.net/book/the-book-2nd.html>`_


Monte Carlo
-----------

These kinds of algorithms play each episode to its end and then perform updates
based on rewards gathered over the course of the episode. This type of
algorithm is covered in Chapter 5 of `Sutton & Barto
<http://incompleteideas.net/book/the-book-2nd.html>`_. Also have a look at the
following Jupyter notebooks for some simple examples:

    - :doc:`../notebooks/blackjack-linear-model-monte-carlo`
    - :doc:`../notebooks/frozenlake-linear-model-montecarloq-and-reinforce`


TD(0)
-----

These are temporal difference type algorithms, where we use a bootstrapped
target over a single timestep. These algorithms are covered in Chapter 6 of
`Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_. Here's
an example notebook:

    - :doc:`../notebooks/cartpole-linear-model-td0`


n-step Bootstrap
----------------

These kinds of algorithms interpolate between `Monte Carlo`_ and `TD(0)`_. It's
kind of like a truncated Monte Carlo algorithm, where :math:`n` is the maximum
nummber of allowed timesteps. Once this cut-off is reached, we bootstrap to
estimate the return for the remainder of the episode, similar to TD(0). See
Chapter 7 of `Sutton & Barto <http://incompleteideas.net/book/the-book-
2nd.html>`_. An example notebook:

    - :doc:`../notebooks/cartpole-linear-model-nstep-bootstrap`


Policy Gradient
---------------

These are algorithms that update the policy directly. The underlying object
that is updated is an updateable policy object, cf. :mod:`keras_gym.policies`.

The most basic example is the implementation of the REINFORCE algorithm:
:class:`Reinforce <keras_gym.algorithms.Reinforce>`.

Another very broad class of algorithms is implemented in
:class:`AdvantageActorCritic <keras_gym.algorithms.AdvantageActorCritic>`,
which approximates the advantage function not by sampling (as in REINFORCE) but
by learning an auxiliary value function :math:`V(s)`. The implementation is not
an algorithm on its own. Instead, :class:`AdvantageActorCritic
<keras_gym.algorithms.AdvantageActorCritic>` allows you to bundle the policy
(actor) with the value function (critic). The combined actor-critic object can
then be passed to an ordinary V-type algorithm like :class:`MonteCarloV
<keras_gym.algorithms.MonteCarloV>`, :class:`ValueTD0
<keras_gym.algorithms.ValueTD0>`, :class:`NStepBootstrap
<keras_gym.algorithms.NStepBootstrap>`, etc.

.. code:: python

    import gym
    from keras_gym.predefined.linear import LinearV, LinearSoftmaxPolicy
    from keras_gym.algorithms import AdvantageActorCritic, NStepBootstrap

    env = gym.make(...)

    actor_critic = AdvantageActorCritic(
        policy=LinearSoftmaxPolicy(env, lr=0.01),
        value_function=LinearV(env, lr=0.1))

    a2c_algo = NStepBootstrap(actor_critic, n=10)

    # usual boilerplate to run env episodes
    ...

    # within an episode, update the actor-critic by feeding a transition
    a = ac.policy.thompson(s)
    s_next, r, done, info = env.step(a)
    a2c_algo.update(s, a, r, s_next, done)


References
----------

.. toctree::
    :maxdepth: 2
    :glob:

    monte_carlo
    td0
    nstep_bootstrap
    policy_gradient
