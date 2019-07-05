"""
Base class for potential neuroevolution encoding genomes to subclass. This ensures that the encoding-genomes used in the
Tensorflow-Neuroevolution framework implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseGenome(object, metaclass=ABCMeta):

    @abstractmethod
    def to_phenotype(self):
        raise NotImplementedError("Should implement to_phenotype()")

    @abstractmethod
    def add_layer(self, index, layer):
        raise NotImplementedError("Should implement add_layer()")

    @abstractmethod
    def replace_layer(self, index, layer_type, units=None, activation=None):
        raise NotImplementedError("Should implement replace_layer()")

    @abstractmethod
    def get_layer_count(self):
        raise NotImplementedError("Should implement get_layer_count()")

    @abstractmethod
    def set_id(self, genome_id):
        raise NotImplementedError("Should implement set_id()")

    @abstractmethod
    def get_id(self):
        raise NotImplementedError("Should implement get_id()")

    @abstractmethod
    def set_fitness(self, fitness):
        raise NotImplementedError("Should implement set_fitness()")

    @abstractmethod
    def get_fitness(self):
        raise NotImplementedError("Should implement get_fitness()")
