Glossary
========

In this package we make heavy use of function approximators using
:class:`keras.Model` objects. In Section 1 we list the available types of
function approximators. A function approximator uses multiple keras models to
support its full functionality. The different types keras models are listed in
Section 2. Finally, in Section 3 we list the different kinds of inputs and
outputs that our keras models expect.

.. contents::
    :local:


1. Function approximator types
------------------------------

.. glossary::

    function approximator

        A function approximator is any object that can be updated.

    body

        The *body* is what we call the part of the computation graph that may
        be shared between e.g. policy (actor) and value function (critic). It
        is typlically the part of a neural net that does most of the heavy
        lifting. One may think of the :func:`body` as an elaborate automatic
        feature extractor.

    head

        The *head* is the part of the computation graph that actually generates
        the desired output format/shape. As its input, it takes the output of
        :term:`body`. The different heads that :class:`FunctionApproximator
        <keras_gym.FunctionApproximator>` class provides are:

        **head_v**

            This is the :term:`state value <state value function>` head. It
            returns a scalar V-value :math:`v(s)\in\mathbb{R}`.

        **head_q1**

            This is the :term:`type-I Q-value <type-I state-action value
            function>` head. It returns a scalar Q-value
            :math:`q(s,a)\in\mathbb{R}`.

        **head_q2**

            This is the :term:`type-II Q-value <type-II state-action value
            function>` head. It returns a vector of Q-values
            :math:`q(s,.)\in\mathbb{R}^n`.


        **head_pi**

            This is the policy head. It returns :term:`logits <Z>`
            :math:`z\in\mathbb{R}^n`, i.e. not probabilities. Use a softmax to
            turn the output into probabilities.

    forward_pass

        This is just the consecutive application of :term:`head` after
        :term:`body`.


In this package we have four distinct types of :term:`function approximators
<function approximator>`:

.. glossary::

    state value function

        State value functions :math:`v(s)` are implemented by :class:`V
        <keras_gym.V>`.

    type-I state-action value function

        This is the standard state-action value function :math:`q(s,a)`. It
        models the Q-function as

        .. math::

            (s, a) \mapsto q(s,a)\ \in\ \mathbb{R}

        This function approximator is implemented by :class:`QTypeI
        <keras_gym.QTypeI>`.

    type-II state-action value function

        This type of state-action value function is different from type-I in
        that it models the Q-function as

        .. math::

            s \mapsto q(s,.)\ \in\ \mathbb{R}^n

        where :math:`n` is the number of actions. The type-II Q-function is
        implemented by :class:`QTypeII <keras_gym.QTypeII>`.

    updateable policy

        This function approximator represents a policy directly. It is
        implemented by e.g. :class:`SoftmaxPolicy
        <keras_gym.SoftmaxPolicy>`.

    actor-critic

        This is a special function approximator that allows for the sharing of
        parts of the computation graph between a value function (critic) and a
        policy (actor).


.. note::

    At the moment, :term:`type-II Q-functions <type-II state-action value
    function>` and :term:`updateable policies <updateable policy>` are only
    implemented for environments with a :class:`Discrete <gym.spaces.Discrete>`
    action space.



2. Keras model types
--------------------

Now each function approximator takes multiple :class:`keras.Model` objects. The
different models are named according to role they play in the functions
approximator object:

.. glossary::

    train_model

        This :class:`keras.Model` is used for training.

    predict_model

        This :class:`keras.Model` is used for predicting.

    target_model

        This :class:`keras.Model` is a kind of shadow copy of
        :term:`predict_model` that is used in off-policy methods. For instance,
        in DQN we use it for reducing the variance of the bootstrapped target
        by synchronizing with :term:`predict_model` only periodically.

.. note::

    The specific input depends on the type of function approximator you're
    using. These are provided in each individual class doc.


3. Keras model inputs/outputs
-----------------------------

Each :class:`keras.Model` object expects specific inputs and outputs. These are
provided in each individual function approximator's docs.

Below we list the different available arrays that we might use as
inputs/outputs to our keras models.

.. glossary::

    S

        A batch of (preprocessed) state observations. The shape is
        ``[batch_size, ...]`` where the ellipses might be any number of
        dimensions.

    A

        A batch of actions taken, with shape ``[batch_size]``.

    P

        A batch of action propensities according to the policy
        :math:`\pi(a|s)`.

    G

        A batch of (:math:`\gamma`-discounted) returns, shape:
        ``[batch_size]``.

    Rn

        A batch of partial (:math:`\gamma`-discounted) returns. For instance,
        in n-step bootstrapping these are given by:

        .. math::

            R^{(n)}_t\ =\ R_t + \gamma\,R_{t+1} + \dots +
            \gamma^{n-1}\,R_{t+n-1}

        In other words, it's the part of the n-step return *without* the
        bootstrapping term. The shape is ``[batch_size]``.

    In

        A batch of bootstrap factors. For instance, in n-step bootstrapping
        these are given by :math:`I^{(n)}_t=\gamma^n` when bootstrapping and
        :math:`I^{(n)}_t=0` otherwise. It is used in bootstrapped updates. For
        instance, the n-step bootstrapped target makes use of it as follows:

            .. math::

                G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,Q(S_{t+1}, A_{t+1})

        The shape is ``[batch_size]``.

    S_next

        A batch of (preprocessed) next-state observations. This is typically
        used in bootstrapping (see :term:`In`). The shape is ``[batch_size,
        ...]`` where the ellipses might be any number of dimensions.

    A_next

        A batch of next-actions to be taken. These can be actions that were
        actually taken (on-policy), but they can also be any other would-be
        next-actions (off-policy). The shape is ``[batch_size]``.

    P_next

        A batch of action propensities according to the policy
        :math:`\pi(a|s)`.

    V

        A batch of V-values :math:`v(s)` of shape ``[batch_size]``.

    Q_sa

        A batch of Q-values :math:`q(s,a)` of shape ``[batch_size]``.

    Q_s

        A batch of Q-values :math:`q(s,.)` of shape
        ``[batch_size, num_actions]``.

    Adv

        A batch of advantages :math:`\mathcal{A}(s,a) = q(s,a) - v(s)`, which
        has shape: ``[batch_size]``.

    Z

        A batch of logits, typically used in softmax policies. The expected
        shape is: ``[batch_size, num_actions]``.
