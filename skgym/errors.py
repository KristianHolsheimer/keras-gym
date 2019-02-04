
class ArrayDequeOverflowError(Exception):
    pass


class NoExperienceCacheError(Exception):
    pass


class NoAdversaryError(Exception):
    pass


class UnavailableActionError(Exception):
    pass


class NonDiscreteActionSpaceError(Exception):
    def __init__(self):
        super(NonDiscreteActionSpaceError, self).__init__(
            "I haven't yet implemented continuous action spaces;  please send "
            "me a message to let me know if this is holding you back. -Kris")
