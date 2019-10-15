

class KerasGymError(Exception):
    pass


class ActionSpaceError(KerasGymError):
    pass


class DistributionError(KerasGymError):
    pass


class EpisodeDoneError(KerasGymError):
    pass


class InconsistentCacheInputError(KerasGymError):
    pass


class InsufficientCacheError(KerasGymError):
    pass


class LeafNodeError(KerasGymError):
    pass


class MissingAdversaryError(KerasGymError):
    pass


class MissingModelError(KerasGymError):
    pass


class NotLeafNodeError(KerasGymError):
    pass


class NumpyArrayCheckError(KerasGymError):
    pass


class TensorCheckError(KerasGymError):
    pass


class UnavailableActionError(KerasGymError):
    pass
