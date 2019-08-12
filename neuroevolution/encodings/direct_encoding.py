import tensorflow as tf

from neuroevolution.encodings import BaseGenome
from neuroevolution.encodings import BaseEncoding


class DirectEncodedGenome(BaseGenome):

    def __init__(self):
        pass

    def serialize_genotype(self):
        raise NotImplementedError("Should implement serialize_genotype()")

    def summary(self):
        raise NotImplementedError("Should implement summary()")

    def visualize(self):
        raise NotImplementedError("Should implement visualize()")

    def get_phenotype_model(self):
        raise NotImplementedError("Should implement get_phenotype_model()")

    def get_id(self):
        raise NotImplementedError("Should implement get_id()")

    def set_fitness(self, fitness):
        raise NotImplementedError("Should implement set_fitness()")

    def get_fitness(self):
        raise NotImplementedError("Should implement get_fitness()")


class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        pass
