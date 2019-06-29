

class KerasGymError(Exception):
    pass


class InsufficientCacheError(KerasGymError):
    pass


class EpisodeDoneError(KerasGymError):
    pass


class NonDiscreteActionSpace(KerasGymError):
    pass


class NumpyArrayCheckError(KerasGymError):
    pass


class TensorCheckError(KerasGymError):
    pass


class MissingModelError(KerasGymError):
    pass


class MissingAdversaryError(KerasGymError):
    pass


class UnavailableActionError(KerasGymError):
    pass


class LeafNodeError(KerasGymError):
    pass


class NotLeafNodeError(KerasGymError):
    pass


class InconsistentCacheInputError(KerasGymError):
    pass
