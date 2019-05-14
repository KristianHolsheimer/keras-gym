

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
