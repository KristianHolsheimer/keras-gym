
class KerasGymBaseError(Exception):
    pass


class ArrayDequeOverflowError(KerasGymBaseError):
    pass


class NoExperienceCacheError(KerasGymBaseError):
    pass


class NoAdversaryError(KerasGymBaseError):
    pass


class UnavailableActionError(KerasGymBaseError):
    pass


class BadModelOuputShapeError(KerasGymBaseError):
    def __init__(self, expected_shape, observed_shape):
        super().__init__(
            "expected: {}, but got: {}"
            .format(list(expected_shape), list(observed_shape)))


class NonDiscreteActionSpaceError(KerasGymBaseError):
    def __init__(self):
        super(NonDiscreteActionSpaceError, self).__init__(
            "I haven't yet implemented continuous action spaces;  please send "
            "me a message to let me know if this is holding you back. -Kris")


class ValueBasedPolicyUpdateError(KerasGymBaseError):
    def __init__(self):
        super(ValueBasedPolicyUpdateError, self).__init__(
            "A value-based policy cannot be updated through a policy object; "
            "please update the value function directly.")
