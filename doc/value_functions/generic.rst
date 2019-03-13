Generic Value Functions
=======================

.. autosummary::
    :nosignatures:

    keras_gym.value_functions.GenericV
    keras_gym.value_functions.GenericQ



.. autoclass:: keras_gym.value_functions.GenericV

    .. automethod:: __call__

    .. admonition:: Other Methods

        .. automethod:: X
        .. automethod:: batch_eval
        .. automethod:: update



.. autoclass:: keras_gym.value_functions.GenericQ

    .. automethod:: __call__

    .. admonition:: Other Methods

        .. automethod:: X
        .. automethod:: batch_eval
        .. automethod:: batch_eval_typeI
        .. automethod:: batch_eval_typeII
        .. automethod:: preprocess_typeI
        .. automethod:: preprocess_typeII
        .. automethod:: update
