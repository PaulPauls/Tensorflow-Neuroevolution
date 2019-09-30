from ..base_genome import BaseGenome
from .direct_encoding_model import DirectEncodingModel
from .direct_encoding_visualization import visualize_genome


class DirectEncodingGenome(BaseGenome):
    def __init__(self, genome_id, genotype, trainable, species, generation, dtype, run_eagerly):
        self.genome_id = genome_id
        self.genotype = genotype
        self.trainable = trainable
        self.associated_species = species
        self.originating_generation = generation
        self.dtype = dtype
        self.run_eagerly = run_eagerly

        self.fitness = 0
        self.model = self._create_model()

    def __str__(self):
        string_repr = "DirectEncodingGenome || ID: {:>4} || Fitness: {:>8} || Associated Species: {:>4} || " \
                      "Originating Generation: {:>4} || Gene Count: {:>4}" \
            .format(self.genome_id, self.fitness, self.associated_species, self.originating_generation,
                    len(self.genotype))
        return string_repr

    def serialize(self):
        serialized_genome = {
            'genome_encoding': 'DirectEncodingGenome',
            'genome_id': self.genome_id,
            'fitness': self.fitness,
            'trainable': self.trainable,
            'associated_species': self.associated_species,
            'originating_generation': self.originating_generation,
            'dtype': self.dtype.name,
            'run_eagerly': self.run_eagerly,
            'genotype': [gene.serialize() for gene in self.genotype]
        }
        return serialized_genome

    def visualize(self, view=True, render_file_path=None):
        raise NotImplementedError()

    def get_model(self):
        return self.model

    def get_genotype(self):
        return self.genotype

    def get_topology_levels(self):
        return self.model.topology_dependency_levels

    def get_id(self):
        return self.genome_id

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_associated_species(self, species_id):
        self.associated_species = species_id

    def _create_model(self):
        return DirectEncodingModel(self.genotype, self.trainable, self.dtype, self.run_eagerly)
