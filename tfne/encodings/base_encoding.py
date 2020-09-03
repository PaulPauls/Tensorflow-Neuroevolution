from abc import ABCMeta, abstractmethod

from .base_genome import BaseGenome


class BaseEncoding(object, metaclass=ABCMeta):
    """
    Interface for TFNE compatible encodings, ensuring that those encodings have a function that turns valid genotypes
    into genomes with associated TF models.
    """

    @abstractmethod
    def create_genome(self, *args) -> (int, BaseGenome):
        """
        Create genome from genotype being passed as one or multiple parameters. Return the genome ID as well as the
        newly created genome itself.
        @param args: genome genotype, being one or multiple variables
        @return: tuple of genome ID and newly create genome
        """
        raise NotImplementedError("Subclass of BaseEncoding does not implement 'create_genome()'")

    @abstractmethod
    def serialize(self) -> dict:
        """
        Serialize state of all encoding variables to a json compatible dictionary and return it
        @return: serialized state of the encoding as json compatible dict
        """
        raise NotImplementedError("Subclass of BaseEncoding does not implement 'serialize()'")
