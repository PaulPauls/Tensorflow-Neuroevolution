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
        string_repr = "DirectEncodingGenome || ID: {:>4} || Fitness: {:>4} || Gene Count: {:>4}" \
            .format(self.genome_id, self.fitness, len(self.genotype))
        return string_repr

    def serialize(self):
        serialized_dict = {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "trainable": self.trainable,
            "dtype": str(self.dtype),
            "run_eagerly": self.run_eagerly,
            "genotype": [gene.serialize() for gene in self.genotype]
        }
        return serialized_dict

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
