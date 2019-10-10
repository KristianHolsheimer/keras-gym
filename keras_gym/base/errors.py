

class KerasGymError(Exception):
    pass


class SpaceError(KerasGymError):
    pass


class ActionSpaceError(SpaceError):
    @classmethod
    def feature_request(cls, env):
        return cls(
            "action space of type {0} not yet supported; please file a "
            "feature request on github: "
            "https://github.com/KristianHolsheimer/keras-gym/issues/new?title="
            "Feature%20request%3A%20support%20action%20space%20of%20type%20"
            "{0}".format(env.action_space.__class__.__name__))


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
