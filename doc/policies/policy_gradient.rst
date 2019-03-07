Policy Gradient
===============

.. note::

    I haven't implemented policy gradient algorithms yet, because I first need
    to work through Chapter 13 of Sutton & Barto more carefully.


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


Let's start with the objective functional:[#sumsandintegrals]_

.. math::

    J[\pi]\ =\ \int ds\,da\,\pi(a|s)\,\mu(s)\,\mathcal{A}(s,a)


where :math:`\mathcal{A}(s,a)=Q(s,a) - V(s)` is the advantage function. The
functional derivative w.r.t. the log-policy is:

.. math::

    \frac{\delta J}{\delta\ln\pi}(s,a)\ =\ \mu(s)\,\pi(a|s)\,\mathcal{A}(s,a)


This suggests that the gradient ascent update for the log-policy becomes:

.. math::

    \Delta\ln\pi(a|s)\ =\ \alpha\,\pi(a|s)\,\mu(s)\,\mathcal{A}(s,a)

Here :math:`\mu(s)` is the density of the state :math:`s`. From a frequentist's
perspective, it tells us how many times we get to observe state :math:`s=S_t`,
relative to all other states. This means that the product
:math:`\pi(a|s)\,\mu(s)` gives us the joint probability for the state-action
pair :math:`(s,a)`. Thus, the way this translates to online updates for a
specific state is simply:

.. math::

    \Delta\ln\pi(a|S_t)\ =\ \alpha\,\pi(a|S_t)\,\mathcal{A}(S_t,a)

If we want to go one step further and sample the state as well, the update for
the specific sampled action :math:`a=A_t` is:

.. math::

    \Delta\ln\pi(A_t|S_t)\ =\ \alpha\,\mathcal{A}(S_t,A_t)

We can satisfy preservation of normalization by picking :math:`V(s)=\int
da\,\pi(a|s)Q(s,a)`, which can be seen via Jensen's inequality, see `Appendix`_
below.


We then use a linear approximation to get the "ground-truth" target we're
interested in, :math:`y=\pi(a|S_t)(1+ \Delta\ln\pi(a|S_t))`. For instance, for
a discrete actions space :math:`a=0,1,\dots`, the labeled example that we feed
to our scikit-learn function approximator :math:`\hat\pi(a|s,\theta)` is:

.. math::
    :label: bootstrap

    X\ &=\ \phi(S_t)\\
    \qquad y_a\ &=\ \hat\pi(a|S_t,\theta)\left(1 + \alpha\,\hat\pi(a|S_t,\theta)\mathcal{A}(S_t,a)\right)

where :math:`\phi` is some feature preprocessor (called a :term:`transformer`
throughout this package). Just like before, we could also just sample the
action :math:`a=A_t`, such that

.. math::
    :label: bootstrap_sampled

    X\ &=\ \phi(S_t)\\
    \qquad y_{A_t}\ &=\ \hat\pi(A_t|S_t,\theta)\left(1 + \alpha\mathcal{A}(S_t,A_t)\right)



Essentially, what we're doing is gradient boosting for our log-policy. In fact,
if we were to do actual gradient boosting, we would approximate the advantage
function (the functional gradient) and then tune :math:`\alpha` to optimize
objective. The updates would be:

.. math::

    \hat{Q}(s,a)\ &\approx\ Q(s,a) \\
    \ln\hat{\pi}(a|s)\ &\leftarrow\ \ln\hat{\pi}(a|s) + \alpha \frac{\delta J}{\delta\ln\pi}(s,a)


In contrast to gradient boosting, in which we would keep the entire sequence of
learners, we iteratively update our target relative to our previous prediction
as per Eq. :eq:`bootstrap`.


If you're reading this and you think I'm making a mistake somewhere, please
would you let me know? I would really appreciate it!


.. rubric:: Footnotes

.. [#sumsandintegrals]

    For discrete action/state spaces, we can replace the integrals by sums,
    e.g. :math:`\int da \to \sum_a`.


Appendix
--------

To see that :math:`V(s)=\int da\,\pi(a|s)Q(s,a)` indeed preserves
normalization under updates of the form

.. math::

    \ln\pi(a|s)\ \leftarrow\ \ln\pi(a|s) + \Delta\ln\pi(a|s)

consider the following application of Jensen's inequality:

.. math::

    1\ &=\ \int da\,\pi(a|s) \\
     \ &=\ \int da\,\exp\left( \ln\pi(a|s) \right) \\
     \ &\equiv\ \int da\,\exp\left( \ln\pi(a|s) + \Delta\ln\pi(a|s) \right) \\
     \ &=\ \int da\,\pi(a|s)\,\exp\left( \Delta\ln\pi(a|s) \right) \\
     \ &=\ \mathbb{E}_\pi\left[\exp\left(\Delta\ln\pi(A|s)\right)\right] \\
     \ &\geq\ \exp\mathbb{E}_\pi\left[\Delta\ln\pi(A|s)\right]

We saturate Jensen's lower bound at the point where

.. math::

    0\ &=\ \mathbb{E}_\pi\left[\Delta\ln\pi(A|s)\right]\\
     \ &=\ \int da\,\pi(a|s)\,\Delta\ln\pi(A|s) \\
     \ &=\ \alpha\int da\,\pi(a|s)\,\mathcal{A}(s,a) \\
     \ &=\ \alpha\int da\,\pi(a|s)\,\left(Q(s,a) - V(s)\right)

This is guaranteed when we pick :math:`V(s)=\int da\,\pi(a|s)Q(s,a)`.
