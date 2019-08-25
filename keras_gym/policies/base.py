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
    def dist_params(self, s):
        """
        Get the parameters of the (conditional) probability distribution
        :math:`\\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation.

        Returns
        -------
        params : nd array

            An array containing the distribution parameters.

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
