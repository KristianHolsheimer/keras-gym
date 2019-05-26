from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """
    Abstract base class for policy objects.

    """
    @abstractmethod
    def __call__(self, s):
        """
        Draw an action from the current policy :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a : action

            A single action proposed under the current policy.

        """
        pass

    @abstractmethod
    def proba(self, s):
        """
        Get the probabilities over all actions :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        pi : 1d array, shape: [num_actions]

            Probabilities over all actions.

            **Note.** This hasn't yet been implemented for non-discrete action
            spaces.

        """
        pass

    @abstractmethod
    def greedy(self, s):
        """
        Draw the greedy action, i.e. :math:`\\arg\\max_a\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        a : action

            A single action proposed under the current policy.

        """
        pass
