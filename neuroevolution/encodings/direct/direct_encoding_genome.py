from ..base_genome import BaseGenome
from .direct_encoding_model import DirectEncodingModel
from .direct_encoding_visualization import visualize_genome


class DirectEncodingGenome(BaseGenome):
    def __init__(self, genome_id, genotype, trainable, dtype, run_eagerly):
        self.genome_id = genome_id
        self.genotype = genotype
        self.trainable = trainable
        self.dtype = dtype
        self.run_eagerly = run_eagerly

        self.fitness = 0
        self.model = self._create_model()

    def __str__(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def visualize(self, view=True, render_file_path=None):
        raise NotImplementedError()

    def get_model(self):
        return self.model

    def get_genotype(self):
        return self.genotype

    def get_id(self):
        return self.genome_id

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def _create_model(self):
        return DirectEncodingModel(self.genotype, self.trainable, self.dtype, self.run_eagerly)
