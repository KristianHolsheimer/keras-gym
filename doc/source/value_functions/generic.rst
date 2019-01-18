Generic Value Functions
=======================

.. autosummary::
    :nosignatures:

    skgym.value_functions.GenericV
    skgym.value_functions.GenericQ
    skgym.value_functions.GenericQTypeI
    skgym.value_functions.GenericQTypeII


.. automodule:: skgym.value_functions

    .. autoclass:: GenericV
        :members: __call__, X, update, batch_eval

    .. autoclass:: GenericQ
        :members: __call__, X, update, preprocess_typeI, preprocess_typeII, batch_eval_typeI, batch_eval_typeII

    .. autoclass:: GenericQTypeI
        :members: __call__, X, update, preprocess_typeI, preprocess_typeII, batch_eval_typeI, batch_eval_typeII

    .. autoclass:: GenericQTypeII
        :members: __call__, X, update, preprocess_typeI, preprocess_typeII, batch_eval_typeI, batch_eval_typeII
