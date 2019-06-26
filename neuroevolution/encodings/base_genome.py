"""
Base class for potential neuroevolution encoding genomes to subclass. This ensures that the encoding-genomes used in the
Tensorflow-Neuroevolution framework implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseGenome(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def to_phenotype(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement to_phenotype()")

    @abstractmethod
    def get_id(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_id()")

    @abstractmethod
    def get_fitness(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_fitness()")
