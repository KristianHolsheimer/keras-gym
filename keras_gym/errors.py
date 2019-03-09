
class keras_gymBaseError(Exception):
    pass


class ArrayDequeOverflowError(keras_gymBaseError):
    pass


class NoExperienceCacheError(keras_gymBaseError):
    pass


class NoAdversaryError(keras_gymBaseError):
    pass


class UnavailableActionError(keras_gymBaseError):
    pass


class NonDiscreteActionSpaceError(keras_gymBaseError):
    def __init__(self):
        super(NonDiscreteActionSpaceError, self).__init__(
            "I haven't yet implemented continuous action spaces;  please send "
            "me a message to let me know if this is holding you back. -Kris")


class ValueBasedPolicyUpdateError(keras_gymBaseError):
    def __init__(self):
        super(ValueBasedPolicyUpdateError, self).__init__(
            "A value-based policy cannot be updated through a policy object; "
            "please update the value function directly.")
