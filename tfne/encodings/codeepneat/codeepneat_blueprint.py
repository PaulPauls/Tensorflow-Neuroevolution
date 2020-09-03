import os
import tempfile
from copy import deepcopy

import tensorflow as tf
from graphviz import Digraph


class CoDeepNEATBlueprintNode:
    """
    Node class of the CoDeepNEAT blueprint graph, holding all relevant information for a graph node.
    """

    def __init__(self, gene_id, node, species):
        self.gene_id = gene_id
        self.node = node
        self.species = species


class CoDeepNEATBlueprintConn:
    """
    Connection class of the CoDeepNEAT blueprint graph, holding all relevant information for a graph connection.
    """

    def __init__(self, gene_id, conn_start, conn_end, enabled=True):
        self.gene_id = gene_id
        self.conn_start = conn_start
        self.conn_end = conn_end
        self.enabled = enabled

    def set_enabled(self, enabled):
        """"""
        self.enabled = enabled


class CoDeepNEATBlueprint:
    """
    Blueprint class of the CoDeepNEAT algorithm. Holding information about the evolutionary process for the BP, the
    BP graph as well as the associated optimizer.
    """

    def __init__(self,
                 blueprint_id,
                 parent_mutation,
                 blueprint_graph,
                 optimizer_factory):
        """
        Initiate CoDeepNEAT blueprint, setting its fitness to 0 and processing the supplied blueprint graph
        @param blueprint_id: int, ID of blueprint
        @param parent_mutation: dict summarizing the parent mutation for the BP
        @param blueprint_graph: dict of the blueprint graph, associating graph gene ID with graph gene, being either
                                a BP graph node or a BP graph connection.
        @param optimizer_factory: instance of a configured optimizer factory that produces configured TF optimizers
        """
        # Register parameters
        self.blueprint_id = blueprint_id
        self.parent_mutation = parent_mutation
        self.blueprint_graph = blueprint_graph
        self.optimizer_factory = optimizer_factory

        # Initialize internal variables
        self.fitness = 0

        # Declare graph related internal variables
        # species: set of all species present in blueprint
        # node_species: dict mapping of each node to its corresponding species
        # node dependencies: dict mapping of nodes to the set of node upon which they depend upon
        # graph topology: list of sets of dependency levels, with the first set being the nodes that depend on nothing,
        #                 the second set being the nodes that depend on the first set, and so on
        self.species = set()
        self.node_species = dict()
        self.node_dependencies = dict()
        self.graph_topology = list()

        # Process graph to set graph related internal variables
        self._process_graph()

    def _process_graph(self):
        """
        Process graph and save the results to the following internal variables:
        species: set of all species present in blueprint
        node_species: dict mapping of each node to its corresponding species
        node dependencies: dict mapping of nodes to the set of node upon which they depend upon
        graph topology: list of sets of dependency levels, with the first set being the nodes that depend on nothing,
                        the second set being the nodes that depend on the first set, and so on
        """
        # Create set of species (self.species, set), assignment of nodes to their species (self.node_species, dict) as
        # well as the assignment of nodes to the nodes they depend upon (self.node_dependencies, dict)
        for gene in self.blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                self.node_species[gene.node] = gene.species
                self.species.add(gene.species)
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn):
                # Only consider a connection for the processing if it is enabled
                if gene.conn_end in self.node_dependencies:
                    self.node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    self.node_dependencies[gene.conn_end] = {gene.conn_start}
        # Remove the 'None' species assigned to Input node
        self.species.remove(None)

        # Topologically sort the graph and save into self.graph_topology as a list of sets of levels, with the first
        # set being the layer dependent on nothing and the following sets depending on the values of the preceding sets
        node_deps = self.node_dependencies.copy()
        node_deps[1] = set()  # Add Input node 1 to node dependencies as dependent on nothing
        while True:
            # find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            if not dependencyless:
                # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
                # occured
                if node_deps:
                    raise ValueError("Circular Dependency Error when sorting the topology of the Blueprint graph.\n"
                                     "Parent mutation: {}".format(self.parent_mutation))
                # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
                # regularly
                break
            # Add dependencyless nodes of current generation to list
            self.graph_topology.append(dependencyless)

            # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
            # dependencies of other nodes in order to create next iteration
            for node in dependencyless:
                del node_deps[node]
            for node, dep in node_deps.items():
                node_deps[node] = dep - dependencyless

    def __str__(self) -> str:
        """
        @return: string representation of the blueprint
        """
        return "CoDeepNEAT Blueprint | ID: {:>6} | Fitness: {:>6} | Nodes: {:>4} | Module Species: {} | Optimizer: {}" \
            .format('#' + str(self.blueprint_id),
                    self.fitness,
                    len(self.node_species),
                    self.species,
                    self.optimizer_factory.get_name())

    def visualize(self, show=True, save_dir_path=None) -> str:
        """
        Visualize the blueprint. If 'show' flag is set to true, display the blueprint after rendering. If
        'save_dir_path' is supplied, save the rendered blueprint as file to that directory. Return the saved file path
        as string.
        @param show: bool flag, indicating whether the rendered blueprint should be displayed or not
        @param save_dir_path: string of the save directory path the rendered blueprint should be saved to.
        @return: string of the file path to which the rendered blueprint has been saved to
        """
        # Check if save_dir_path is supplied and if it is supplied in the correct format. If not correct format or
        # create a new save_dir_path. Ensure that the save_dir_path exists by creating the directories.
        if save_dir_path is None:
            save_dir_path = tempfile.gettempdir()
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        os.makedirs(save_dir_path, exist_ok=True)

        # Set filename and save file path as the blueprint id and indicate that its the graph being plotted
        filename = f"blueprint_{self.blueprint_id}_graph"

        # Create Digraph, setting name and graph orientaion
        dot = Digraph(name=filename, graph_attr={'rankdir': 'TB'})

        # Traverse all bp graph genes, adding the nodes and edges to the graph
        for gene in self.blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                label = f"Node: {gene.node}"
                if gene.node != 1:
                    label += f"\nSpecies: {gene.species}"
                dot.node(name=str(gene.node), label=label)
            elif gene.enabled:
                dot.edge(str(gene.conn_start), str(gene.conn_end))

        # Highlight Input and Output Nodes
        with dot.subgraph(name='cluster_1') as dot_in:
            dot_in.node('1')
            dot_in.attr(label='inputs', color='blue')
        with dot.subgraph(name='cluster_2') as dot_out:
            dot_out.node('2')
            dot_out.attr(label='outputs', color='grey')

        # Render created dot graph, optionally showing it
        dot.render(filename=filename, directory=save_dir_path, view=show, cleanup=True, format='svg')

        # Return the file path to which the blueprint graph plot was saved
        return save_dir_path + f"{filename}.svg"

    def calculate_gene_distance(self, other_bp) -> float:
        """
        Calculate the distance between 2 blueprint graphs by determining it as the congruence of genes and subtracting
        it from the maximum distance of 1
        @param other_bp: blueprint to which the gene distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """
        # Calculate the gene distance betweeen 2 blueprints by calculating the congruence of genes and subtracting it
        # from the maximum distance of 1
        bp_node_ids = set(self.blueprint_graph.keys())
        other_bp_node_ids = set(other_bp.blueprint_graph.keys())
        node_id_intersection = bp_node_ids.intersection(other_bp_node_ids)

        intersection_length = len(node_id_intersection)
        gene_congruence = (intersection_length / len(bp_node_ids) + intersection_length / len(other_bp_node_ids)) / 2.0

        return 1.0 - gene_congruence

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        @return: TF optimizer instance from the associated and pre-configured optimizer factory
        """
        return self.optimizer_factory.create_optimizer()

    def copy_parameters(self) -> ({int: object}, object):
        """
        @return: deepcopied blueprint graph and optimizer factory
        """
        return deepcopy(self.blueprint_graph), self.optimizer_factory.duplicate()

    def update_blueprint_graph(self):
        """
        Reset graph related internal variables and reprocess graph. Necessary if bp_graph has been updated
        """
        self.species = set()
        self.node_species = dict()
        self.node_dependencies = dict()
        self.graph_topology = list()
        self._process_graph()

    def serialize(self) -> dict:
        """
        Serialize all blueprint variables to a json compatible dictionary and return it
        @return: serialized blueprint variables as json compatible dict
        """
        # Create serialization of the blueprint graph suitable for json output
        serialized_blueprint_graph = dict()
        for gene_id, gene in self.blueprint_graph.items():
            serialized_gene = dict()
            if isinstance(gene, CoDeepNEATBlueprintNode):
                serialized_gene['node'] = gene.node
                serialized_gene['species'] = gene.species
            else:
                serialized_gene['conn_start'] = gene.conn_start
                serialized_gene['conn_end'] = gene.conn_end
                serialized_gene['enabled'] = gene.enabled
            serialized_blueprint_graph[gene_id] = serialized_gene

        return {
            'blueprint_type': 'CoDeepNEAT',
            'blueprint_id': self.blueprint_id,
            'parent_mutation': self.parent_mutation,
            'blueprint_graph': serialized_blueprint_graph,
            'optimizer_factory': self.optimizer_factory.get_parameters()
        }

    def set_fitness(self, fitness):
        """"""
        self.fitness = fitness

    def get_id(self) -> int:
        """"""
        return self.blueprint_id

    def get_fitness(self) -> float:
        """"""
        return self.fitness

    def get_blueprint_graph(self) -> {int: object}:
        """"""
        return self.blueprint_graph

    def get_species(self) -> {int, ...}:
        """"""
        return self.species

    def get_node_species(self) -> {int: int}:
        """"""
        return self.node_species

    def get_node_dependencies(self) -> {int: {int, ...}}:
        """"""
        return self.node_dependencies

    def get_graph_topology(self) -> [{int, ...}, ...]:
        """"""
        return self.graph_topology
