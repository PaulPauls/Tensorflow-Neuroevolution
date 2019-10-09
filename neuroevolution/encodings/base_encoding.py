from abc import ABCMeta, abstractmethod

from .base_genome import BaseGenome


class BaseEncoding(object, metaclass=ABCMeta):

    @abstractmethod
    def create_genome(self, genotype) -> (int, BaseGenome):
        """
        Create genome based on the supplied genotype, with continuous genome-id for each newly created genome
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :return: tuple of continuous genome-id and created genome
        """
        raise NotImplementedError("Should implement create_genome()")
