"""
Base class for potential neuroevolution algorithms to subclass. This ensures that the ne-algorithms used in
Tensorflow-Neuroevolution implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement mutate_population()")

    @abstractmethod
    def select_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement mutate_population()")

    @abstractmethod
    def mutate_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement mutate_population()")

    @abstractmethod
    def speciate_genomes(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement mutate_population()")
