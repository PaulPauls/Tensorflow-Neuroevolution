from copy import deepcopy

import tensorflow as tf


class CoDeepNEATBlueprintNode:
    """"""

    def __init__(self, gene_id, node, species):
        self.gene_id = gene_id
        self.node = node
        self.species = species


class CoDeepNEATBlueprintConn:
    """"""

    def __init__(self, gene_id, conn_start, conn_end):
        self.gene_id = gene_id
        self.conn_start = conn_start
        self.conn_end = conn_end
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled


class CoDeepNEATBlueprint:
    """"""

    def __init__(self,
                 blueprint_id,
                 parent_mutation,
                 blueprint_graph,
                 optimizer_factory):
        """"""
        # Register parameters
        self.blueprint_id = blueprint_id
        self.parent_mutation = parent_mutation
        self.blueprint_graph = blueprint_graph
        self.optimizer_factory = optimizer_factory

        # Initialize internal variables
        self.fitness = None

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
        """"""
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
        """"""
        return "CoDeepNEAT Blueprint | ID: {:>6} | Fitness: {:>6} | Nodes: {:>4} | Module Species: {} | Optimizer: {}" \
            .format('#' + str(self.blueprint_id),
                    self.fitness,
                    len(self.node_species),
                    self.species,
                    self.optimizer_factory.get_name())

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer_factory.create_optimizer()

    def copy_parameters(self) -> ({int: object}, object):
        """"""
        return (deepcopy(self.blueprint_graph),
                self.optimizer_factory.duplicate())

    def serialize(self) -> dict:
        """"""
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
            'blueprint_id': self.blueprint_id,
            'parent_mutation': self.parent_mutation,
            'blueprint_graph': serialized_blueprint_graph,
            'optimizer_factory': self.optimizer_factory.get_parameters()
        }

    def set_fitness(self, fitness):
        self.fitness = fitness

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

    def get_id(self) -> int:
        return self.blueprint_id

    def get_fitness(self) -> float:
        return self.fitness
