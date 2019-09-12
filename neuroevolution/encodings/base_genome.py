from abc import ABCMeta, abstractmethod


class BaseGenome(object, metaclass=ABCMeta):
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
    def get_weights(self):
        raise NotImplementedError("Should implement get_weights()")

    @abstractmethod
    def get_genotype(self):
        raise NotImplementedError("Should implement get_genotype()")

    @abstractmethod
    def get_id(self):
        raise NotImplementedError("Should implement get_id()")

    @abstractmethod
    def set_weights(self, weights):
        raise NotImplementedError("Should implement set_weights()")

    @abstractmethod
    def set_fitness(self, fitness):
        raise NotImplementedError("Should implement set_fitness()")

    @abstractmethod
    def get_fitness(self):
        raise NotImplementedError("Should implement get_fitness()")
