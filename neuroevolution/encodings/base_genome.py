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
    def add_layer(self, index, layer):
        """
        ToDo
        :param index:
        :param layer:
        :return:
        """
        raise NotImplementedError("Should implement add_layer()")

    @abstractmethod
    def replace_layer(self, index, layer_type, units=None, activation=None):
        """
        ToDo
        :param index:
        :param layer_type:
        :param units:
        :param activation:
        :return:
        """
        raise NotImplementedError("Should implement replace_layer()")

    @abstractmethod
    def get_layer_count(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_layer_count()")

    @abstractmethod
    def set_id(self, genome_id):
        """
        ToDo
        :param genome_id:
        :return:
        """
        raise NotImplementedError("Should implement set_id()")

    @abstractmethod
    def get_id(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_id()")

    @abstractmethod
    def set_fitness(self, fitness):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement set_fitness()")

    @abstractmethod
    def get_fitness(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_fitness()")
