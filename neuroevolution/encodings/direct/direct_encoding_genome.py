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
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()
        '''
        graph_name = "Graph_of_Genome_{}".format(genome.genome_id)

        dot = Digraph(name=graph_name, format='png')
        dot.attr(rankdir='BT')

        edge_list = list()
        gene = genome.genotype
        while gene:
            edge = ('{}'.format(gene.conn_in), '{}'.format(gene.conn_out))
            edge_list.append(edge)
            gene = gene.next_gene

        dot.edges(edge_list)

        with dot.subgraph(name='cluster_1') as dot_in:
            for node in genome.inputs_outputs['inputs']:
                dot_in.node('{}'.format(node))
            dot_in.attr(label='inputs')
            dot_in.attr(color='blue')

        with dot.subgraph(name='cluster_2') as dot_out:
            for node in genome.inputs_outputs['outputs']:
                dot_out.node('{}'.format(node))
            dot_out.attr(label='outputs')
            dot_out.attr(color='grey')

        dot.view(tempfile.mktemp())
        '''

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
        phenotype_model = DirectEncodingModel(self.genotype, self.activations, self.trainable)
        topology_levels = phenotype_model.get_topology_dependency_levels()
        return phenotype_model, topology_levels
