from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def __call__(self, s):
        pass

    @abstractmethod
    def proba(self, s):
        pass

    @abstractmethod
    def greedy(self, s):
        pass
