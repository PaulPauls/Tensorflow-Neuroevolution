import tempfile
from graphviz import Digraph


def visualize_direct_encoding_genome(genome):
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
