
class SkGymBaseError(Exception):
    pass


class ArrayDequeOverflowError(SkGymBaseError):
    pass


class NoExperienceCacheError(SkGymBaseError):
    pass


class NoAdversaryError(SkGymBaseError):
    pass


class UnavailableActionError(SkGymBaseError):
    pass


class NonDiscreteActionSpaceError(SkGymBaseError):
    def __init__(self):
        super(NonDiscreteActionSpaceError, self).__init__(
            "I haven't yet implemented continuous action spaces;  please send "
            "me a message to let me know if this is holding you back. -Kris")


class ValueBasedPolicyUpdateError(SkGymBaseError):
    def __init__(self):
        super(ValueBasedPolicyUpdateError, self).__init__(
            "A value-based policy cannot be updated through a policy object; "
            "please update the value function directly.")
