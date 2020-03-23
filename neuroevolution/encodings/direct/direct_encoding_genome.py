import typing
from ..base_genome import BaseGenome
from .direct_encoding_model import DirectEncodingModelTrainable, DirectEncodingModelNontrainable
from .direct_encoding_visualization import visualize_genome


class DirectEncodingGenome(BaseGenome):
    """
    Implementation of a DirectEncoding genome as employed by NE-algorithms like NEAT. DirectEncoding genomes have each
    connection, connection-weight, node, node-bias, etc of the corresponding Tensorflow model explicitely encoded in
    their genotype, which is made up of DirectEncoding genes. Upon creation does the DirectEncoding genome immediately
    create the phenotype Tensorflow model based on the genotype.
    """

    def __init__(self, genome_id, genotype, trainable, dtype, run_eagerly):
        """
        Set ID and genotype of genome to the supplied parameters, set the default fitness value of the genome to 0 and
        create the Tensorflow model phenotype using the supplied genotype, trainable, dtype and run_eagerly parameters
        and save it as the model.
        """
        self.genome_id = genome_id
        self.genotype = genotype
        self.fitness = 0

        if trainable:
            self.model = DirectEncodingModelTrainable(genotype, dtype, run_eagerly)
        else:
            self.model = DirectEncodingModelNontrainable(genotype, dtype)

    def __str__(self) -> str:
        string_repr = "DirectEncodingGenome || ID: {:>4} || Fitness: {:>8} || Gene Count: {:>4}" \
            .format(self.genome_id, self.fitness, len(self.genotype))
        return string_repr

    def visualize(self, view=True, filename=None, render_dir_path=None):
        """
        Display rendered genome or save rendered genome to specified path or do both
        :param view: flag if rendered genome should be displayed
        :param filename: string of filename of the visualization render, excluding the extension
        :param render_dir_path: string of directory path, specifying where the genome render should be saved
        """
        visualize_genome(self.genome_id, self.genotype, self.model.topology_levels, view, filename, render_dir_path)

    def serialize(self) -> dict:
        """
        Shallow serializes genome and returns it as a dict. Serialization is shallow as only genome characteristics are
        serialized that are required to recreate genome from scratch - but no internal states(fitness, dtype, etc).
        :return: dict; dict containing the shallow serialization of the genome
        """
        return {
            'genome_type': 'DirectEncodingGenome',
            'genotype': [gene.serialize() for gene in self.genotype.values()]
        }

    def get_model(self) -> typing.Union[DirectEncodingModelTrainable, DirectEncodingModelNontrainable]:
        """
        :return: Tensorflow model phenotype translation of the genome genotype
        """
        return self.model

    def get_genotype(self) -> dict:
        """
        :return: genome genotype dict with the keys being the gene-ids and the values being the genes
        """
        return self.genotype

    def get_topology_levels(self) -> [set]:
        """
        :return: list of topologically sorted sets of nodes. Each list element contains the set of nodes that have to be
                 precomputed before the next list element set of nodes can be computed.
        """
        return self.model.topology_levels

    def get_gene_ids(self) -> []:
        """
        :return: list of gene-ids contained in the genome genotype
        """
        return self.genotype.keys()

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
