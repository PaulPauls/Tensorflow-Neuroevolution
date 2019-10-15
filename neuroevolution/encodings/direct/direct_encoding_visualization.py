import tempfile
from graphviz import Digraph


def visualize_genome(genome_id, genotype, topology_levels, view, render_dir_path):
    """
    Create directed graph to visualize supplied genotype and display rendering or save the rendering to specified path
    or do both
    :param genome_id: int; unique identifier for the genome
    :param genotype: genome genotype dict with the keys being the gene-ids and the values being the genes
    :param topology_levels: list of topologically sorted sets of nodes. Each list element contains the set of nodes that
                            have to be precomputed before the next list element set of nodes can be computed.
    :param view: flag if rendered genome should be displayed
    :param render_dir_path: string of directory path, specifying where the genome render should be saved
    """
    filename = "graph_genome_{}".format(genome_id)
    if render_dir_path is None:
        render_dir_path = tempfile.mkdtemp()

    # Create Digraph and set graph orientaion
    dot = Digraph(name=filename)
    dot.attr(rankdir='BT')

    # Go through all genes and add it to the graph, depending on if its a connection or a node
    for gene in genotype.values():
        try:
            dot.edge(str(gene.conn_in), str(gene.conn_out), label=str(round(gene.conn_weight, 2)))
        except AttributeError:
            dot.node(name=str(gene.node), label=str(gene.node) + "\nBias: " + str(round(gene.bias, 2)))

    # Highlight Input and Output Nodes
    with dot.subgraph(name='cluster_1') as dot_in:
        for node in topology_levels[0]:
            dot_in.node(str(node))
        dot_in.attr(label='inputs', color='blue')
    with dot.subgraph(name='cluster_2') as dot_out:
        for node in topology_levels[-1]:
            dot_out.node(str(node))
        dot_out.attr(label='outputs', color='grey')

    dot.render(filename=filename, directory=render_dir_path, view=view, cleanup=True, format='svg')
