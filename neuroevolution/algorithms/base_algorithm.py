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
    def select_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement select_genomes()")

    @abstractmethod
    def recombine_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement recombine_genomes()")

    @abstractmethod
    def mutate_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement mutate_population()")
