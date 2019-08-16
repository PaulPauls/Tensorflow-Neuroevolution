import tempfile
from graphviz import Digraph

from neuroevolution.encodings import BaseGenome
from neuroevolution.encodings.direct import DirectEncodingModel


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
        return self.phenotype_model(*args, **kwargs)

    def __str__(self):
        serialized_genotype, serialized_activations = self.serialize()
        string_repr = "Genome-ID: {} --- Fitness: {} --- Genotype: {} --- Activations: {}".format(
            self.genome_id, self.fitness, serialized_genotype, serialized_activations)
        return string_repr

    def serialize(self):
        # Convert genome into the explicit genotype and activation dicts that can also be supplied directly
        serialized_genotype = dict()
        for gene in self.genotype:
            serialized_genotype[gene.gene_id] = (gene.conn_in, gene.conn_out)

        return serialized_genotype, self.activations

    def summary(self):
        print(self)
        # Possibly print the phenotype.summary() in this function as well

    def visualize(self):
        # Define meta parameters of Digraph
        dot = Digraph(name="Graph_of_genome_{}".format(self.genome_id))
        dot.attr(rankdir='BT')

        # Specify edges of Digraph
        edge_list = list()
        for gene in self.genotype:
            edge = ('{}'.format(gene.conn_in), '{}'.format(gene.conn_out))
            edge_list.append(edge)
        dot.edges(edge_list)

        # Highlight Input and Output Nodes
        with dot.subgraph(name='cluster_1') as dot_in:
            for node in self.topology_levels[0]:
                dot_in.node('{}'.format(node))
            dot_in.attr(label='inputs')
            dot_in.attr(color='blue')
        with dot.subgraph(name='cluster_2') as dot_out:
            for node in self.topology_levels[-1]:
                dot_out.node('{}'.format(node))
            dot_out.attr(label='outputs')
            dot_out.attr(color='grey')

        # Render graph
        # Alternatively use directory='genome_visualizations' when history of visualized genomes requested
        dot.render(directory=tempfile.mkdtemp(), view=True, cleanup=True, format='png')

    def get_phenotype_model(self):
        return self.phenotype_model

    def get_genotype(self):
        return self.genotype

    def get_genotype_and_activations(self):
        return self.genotype, self.activations

    def get_id(self):
        return self.genome_id

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def _create_phenotype_model(self):
        phenotype_model = DirectEncodingModel(self.genotype, self.activations, self.trainable)
        topology_levels = phenotype_model.get_topology_dependency_levels()
        return phenotype_model, topology_levels
