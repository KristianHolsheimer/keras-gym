from abc import ABC, abstractmethod

from ..base.mixins import RandomStateMixin


__all__ = (
    'BaseProbaDist',
)


class BaseProbaDist(ABC, RandomStateMixin):
    @abstractmethod
    def __init__(self, env, *params, **settings):
        pass

    @abstractmethod
    def sample(self, n=1):
        pass

    @abstractmethod
    def log_proba(self, x):
        pass

    @abstractmethod
    def entropy(self):
        pass

    @abstractmethod
    def cross_entropy(self, other):
        pass

    @abstractmethod
    def kl_divergence(self, other):
        pass

    @abstractmethod
    def proba_ratio(self, other, x):
        pass

    def _check_other(self, other):
        if type(other) is not type(self):
            raise TypeError("'other' must be of the same type as 'self'")
