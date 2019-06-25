"""
Base class for potential neuroevolution algorithms to subclass. This ensures that the ne-algorithms used in the
Tensorflow-Neuroevolution framework implements the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement create_initial_population()")

    @abstractmethod
    def create_new_generation(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement create_new_generation()")
