from abc import ABCMeta, abstractmethod


class BaseGenome(object, metaclass=ABCMeta):
    """
    Base class for potential neuroevolution encoding genomes to subclass. This ensures that the encoding-genomes used in the
    Tensorflow-Neuroevolution framework implement the required functions in the intended way.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Should implement serialize_genotype()")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Should implement serialize_genotype()")

    @abstractmethod
    def serialize(self):
        raise NotImplementedError("Should implement serialize()")

    @abstractmethod
    def summary(self):
        raise NotImplementedError("Should implement summary()")

    @abstractmethod
    def visualize(self):
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def get_phenotype_model(self):
        raise NotImplementedError("Should implement get_phenotype_model()")

    @abstractmethod
    def get_genotype(self):
        raise NotImplementedError("Should implement get_genotype()")

    @abstractmethod
    def get_id(self):
        raise NotImplementedError("Should implement get_id()")

    @abstractmethod
    def set_fitness(self, fitness):
        raise NotImplementedError("Should implement set_fitness()")

    @abstractmethod
    def get_fitness(self):
        raise NotImplementedError("Should implement get_fitness()")
