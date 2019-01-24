Policy Gradient
===============



Target Definition
-----------------

In policy-gradient type policies, we update the policy directly rather than
updating some value function. The policy gradient updates are typically phrased
in terms of the parameters :math:`\theta` of some differentiable function
approximator for the policy.

.. math::

    \pi(A|S)\ \approx\ \hat{\pi}(A|S,\theta)

The update is then formulated as

.. math::

    \Delta\theta\ = \alpha\,\left(G - V(S)\right)\,\nabla \log\pi(A|S)

In this package, however, we don't what to tell the function approximator how
to update its parameter. Instead, we just want to tell the function
approximator what to optimize towards. In other words, we only want to specify
the target :math:`y` given some preprocessed input :math:`X = \phi(S)` say.

In order to figure out what this target need to be, let's go back to the
starting point of derivation of the update rule for :math:`\theta`. Let's start
with our objective function:

In
particular, let's look at how the objective function varies when we vary
:math:`\pi`:

.. math::

    \delta J\ =\ \sum_{s,a}\,\mu(s)\,\mathcal{A}(s,a)\,\delta\pi(a|s)


Here, :math:`\mathcal{A}(s,a)=Q(s,a) - V(s)` is the advantage function. Thus,
the functional derivative is just:

.. math::

    \frac{\delta J}{\delta \pi}(S,A)\ =\ \mu(S)\,\mathcal{A}(S,A)


This suggests that the :math:`\pi` update becomes:

.. math::

    \Delta\pi(A|S)\ =\ \alpha\,\mu(S)\,\mathcal{A}(S,A)

Here :math:`\mu(s)` is the density of the state :math:`S=s`. From a
frequentist's perspective, it tells us how many times we get to observe state
:math:`S=s`, relative to all other states. Thus, the way this translates to
online updates for a specific state is simply:

.. math::

    \Delta\pi(A|s)\ =\ \alpha\,\mathcal{A}(s,A)

The notation :math:`\pi(A|s)` is short-hand for :math:`\pi(A|S=s)`, i.e. the
uppercase :math:`S` is the random variable and the lowercase :math:`s` is a
specific value (variate). Note that preservation of normalization fixes
:math:`V(S)=\sum_a\pi(a|s)\mathcal{A}(s,a)`, which follows directly from

.. math::

    0\ =\ \mathbb{E}_\pi\left[\Delta\pi(A|s)\right]
     \ =\ \alpha\sum_a\,\pi(a|s)\,\mathcal{A}(s,a)


We then use a linear approximation to get the "ground-truth" target we're
interested in, :math:`y=\pi(a|s) + \Delta\pi(a|s)`. For instance, for a discrete actions space :math:`a=0,1,\dots`,
the labeled example that we feed to our scikit-learn function approximator is:

.. math::

    X\ &=\ \phi(s)\\
    \qquad y_a\ &=\ \hat{y}_a(s) + \alpha\,\mathcal{A}(s,a)

where :math:`\hat{y}_a(s)=\hat{\pi}(a|s,\theta)` and :math:`\phi` is some
feature preprocessor (called a :term:`transformer` throughout this package).

Of course, we could make the updates more sophisticated, using e.g. momentum
(initialize to :math:`u(s,a)=0`):

.. math::

    u(s,a)\ &\leftarrow\ \eta\,u(s,a) + \alpha\,\mathcal{A}(s,a) \\
    X\ &=\ \phi(s)\\
    \qquad y_a\ &=\ \hat{y}_a(s) + g(s,a)

This does mean, however, that we need one more function approximator to keep
track of policy momentum :math:`u(s,a)`.


If you're reading this and you think I'm making a mistake somewhere, please
would you let me know? I would really appreciate it!
