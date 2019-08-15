from collections import deque

from neuroevolution.encodings import BaseGenome
from neuroevolution.encodings.direct import DirectEncodingModel, visualize_direct_encoding_genome


class DirectEncodingGene:
    def __init__(self, gene_id, conn_in, conn_out):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out


class DirectEncodingGenome(BaseGenome):
    def __init__(self, genome_id, genotype, activations, trainable):
        self.genome_id = genome_id
        self.genotype = genotype
        self.activations = activations
        self.trainable = trainable

        self.fitness = None
        self.phenotype_model, self.topology_levels = self._create_phenotype_model()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()

    def get_phenotype_model(self):
        return self.phenotype_model

    def get_genotype(self):
        return self.genotype, self.activations

    def get_id(self):
        return self.genome_id

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def _create_phenotype_model(self):
        raise NotImplementedError()
