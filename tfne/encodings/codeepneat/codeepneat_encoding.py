from .codeepneat_genome import CoDeepNEATGenome
from .codeepneat_optimizer_factory import OptimizerFactory
from .codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from .modules import CoDeepNEATModuleBase
from tfne.encodings.base_encoding import BaseEncoding

# Import Association dict of the module string name to its implementation class
from .modules import MODULES


class CoDeepNEATEncoding(BaseEncoding):
    """
    CoDeepNEAT encoding that keeps track of created genome, module and blueprint IDs as well as the history of split
    connections.
    """

    def __init__(self, dtype, initial_state=None):
        """
        Create CoDeepNEAT encoding setting the ID counter of CoDeepNEAT elements to 0 and creating default gene id and
        connection split history dicts. If optional initial state is supplied, restore it.
        @param dtype: string of TF dtype with which the genomes and modules will be created
        @param initial_state: optional serialized initial state of the CoDeepNEAT encoding that will be restored
        """
        # Register parameters
        self.dtype = dtype

        # Initialize internal counter variables
        self.genome_id_counter = 0
        self.mod_id_counter = 0
        self.bp_id_counter = 0
        self.bp_gene_id_counter = 0

        # Initialize container that maps a blueprint gene to its assigned blueprint gene id. If a new blueprint gene is
        # created will this container allow to check if that gene has already been created before and has been assigned
        # a unique gene id before. If the blueprint gene has already been created before will the same gene id be used.
        self.gene_to_gene_id = dict()

        # Initialize a counter for nodes and a history container, assigning the tuple of connection start and end a
        # previously assigned node or new node if not yet present in history.
        self.node_counter = 2
        self.conn_split_history = dict()

        # If an initial state is supplied, then the encoding was deserialized. Recreate this initial state.
        if initial_state is not None:
            self.genome_id_counter = initial_state['genome_id_counter']
            self.mod_id_counter = initial_state['mod_id_counter']
            self.bp_id_counter = initial_state['bp_id_counter']
            self.bp_gene_id_counter = initial_state['bp_gene_id_counter']
            self.gene_to_gene_id = initial_state['gene_to_gene_id']
            self.node_counter = initial_state['node_counter']
            self.conn_split_history = initial_state['conn_split_history']

    def create_initial_module(self, mod_type, config_params) -> (int, CoDeepNEATModuleBase):
        """
        Create an initial module by incrementing module ID counter and supplying initial parent_mutation
        @param mod_type: string of the module type that is to be created
        @param config_params: dict of the module parameter range supplied via config
        @return: int of module ID counter and initialized module instance
        """
        # Determine module ID and set the parent mutation to 'init' notification
        self.mod_id_counter += 1
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        return self.mod_id_counter, MODULES[mod_type](config_params=config_params,
                                                      module_id=self.mod_id_counter,
                                                      parent_mutation=parent_mutation,
                                                      dtype=self.dtype,
                                                      self_initialization_flag=True)

    def create_mutated_module(self, parent_module, max_degree_of_mutation) -> (int, CoDeepNEATModuleBase):
        """
        @param parent_module: dict summarizing the mutation of the parent module
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: int of module ID counter and mutated module instance of the same class
        """
        self.mod_id_counter += 1
        return self.mod_id_counter, parent_module.create_mutation(self.mod_id_counter,
                                                                  max_degree_of_mutation)

    def create_crossover_module(self,
                                parent_module_1,
                                parent_module_2,
                                max_degree_of_mutation) -> (int, CoDeepNEATModuleBase):
        """
        Create crossover module calling the crossover function of the fitter parent
        @param parent_module_1: CoDeepNEAT module
        @param parent_module_2: CoDeepNEAT module
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: int of module ID counter and crossed over module instance of the same class
        """
        self.mod_id_counter += 1
        # Determine fitter parent module and call internal crossover function of fitter parent
        if parent_module_1.get_fitness() >= parent_module_2.get_fitness():
            return self.mod_id_counter, parent_module_1.create_crossover(self.mod_id_counter,
                                                                         parent_module_2,
                                                                         max_degree_of_mutation)
        else:
            return self.mod_id_counter, parent_module_2.create_crossover(self.mod_id_counter,
                                                                         parent_module_1,
                                                                         max_degree_of_mutation)

    def create_blueprint_node(self, node, species) -> (int, CoDeepNEATBlueprintNode):
        """
        Create blueprint node. If the node has created before assign it the same gene ID. If the node is novel, create
        a new gene ID.
        @param node: int, specifying the number assigned to the graph node
        @param species: int, specifying the module species with which the module will be replaced in the final genome
                        graph
        @return: int of gene ID and CoDeepNEAT BP graph node instance
        """
        gene_key = (node,)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintNode(bp_gene_id, node, species)

    def create_node_for_split(self, conn_start, conn_end) -> int:
        """
        Determine unique node number based on the connection that is being split. The connection is defined by the start
        and end node. Produce the same new node number for each identical connection that is being split.
        @param conn_start: int of node at which the connection starts
        @param conn_end: int of node at which the connection ends
        @return: New node number for novel connection split, same node number if connection split already happened
        """
        conn_key = (conn_start, conn_end)
        if conn_key not in self.conn_split_history:
            self.node_counter += 1
            self.conn_split_history[conn_key] = self.node_counter

        return self.conn_split_history[conn_key]

    def create_blueprint_conn(self, conn_start, conn_end) -> (int, CoDeepNEATBlueprintConn):
        """
        Create blueprint connection. If the connection (identified by conn_start and conn_end nodes) has already been
        created before assign it the same gene id as before. If the connection is novel, create a new gene id.
        @param conn_start: int of node at which the connection starts
        @param conn_end: int of node at which the connection ends
        @return: int of gene ID and CoDeepNEAT BP graph connection instance
        """
        gene_key = (conn_start, conn_end)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintConn(bp_gene_id, conn_start, conn_end)

    def create_blueprint(self,
                         parent_mutation,
                         blueprint_graph,
                         optimizer_factory) -> (int, CoDeepNEATBlueprint):
        """
        Create blueprint by incrementing blueprint counter and passing Blueprint parameters along
        @param parent_mutation: dict summarizing the parent mutation for the BP
        @param blueprint_graph: dict of the blueprint graph, associating graph gene ID with graph gene, being either
                                a BP graph node or a BP graph connection.
        @param optimizer_factory: instance of a configured optimizer factory that produces configured TF optimizers
        @return: int of blueprint ID and newly created BP instance
        """
        self.bp_id_counter += 1
        return self.bp_id_counter, CoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                       parent_mutation=parent_mutation,
                                                       blueprint_graph=blueprint_graph,
                                                       optimizer_factory=optimizer_factory)

    def create_genome(self,
                      blueprint,
                      bp_assigned_modules,
                      output_layers,
                      input_shape,
                      generation) -> (int, CoDeepNEATGenome):
        """
        Create genome by incrementing genome counter and passing supplied genotype along
        @param blueprint: CoDeepNEAT blueprint instance
        @param bp_assigned_modules: dict associating each BP species with a CoDeepNEAT module instance
        @param output_layers: string of TF deserializable layers serving as output
        @param input_shape: int-tuple specifying the input shape the genome model has to adhere to
        @param generation: int, specifying the evolution generation at which the genome was created
        @return: int of genome ID and newly created CoDeepNEAT genome instance
        """
        self.genome_id_counter += 1
        # Genome genotype: (blueprint, bp_assigned_modules, output_layers)
        return self.genome_id_counter, CoDeepNEATGenome(genome_id=self.genome_id_counter,
                                                        blueprint=blueprint,
                                                        bp_assigned_modules=bp_assigned_modules,
                                                        output_layers=output_layers,
                                                        input_shape=input_shape,
                                                        dtype=self.dtype,
                                                        origin_generation=generation)

    @staticmethod
    def create_optimizer_factory(optimizer_parameters) -> OptimizerFactory:
        """"""
        return OptimizerFactory(optimizer_parameters)

    def serialize(self) -> dict:
        """
        @return: serialized state of the encoding as json compatible dict
        """
        # Convert keys of gene_to_gene_id and conn_split_history dicts to strings as tuples not elligible for json
        # serialization
        serialized_gene_to_gene_id = dict()
        for key, value in self.gene_to_gene_id.items():
            serializable_key = str(key)
            serialized_gene_to_gene_id[serializable_key] = value
        serialized_conn_split_history = dict()
        for key, value in self.conn_split_history.items():
            serializable_key = str(key)
            serialized_conn_split_history[serializable_key] = value

        return {
            'encoding_type': 'CoDeepNEAT',
            'genome_id_counter': self.genome_id_counter,
            'mod_id_counter': self.mod_id_counter,
            'bp_id_counter': self.bp_id_counter,
            'bp_gene_id_counter': self.bp_gene_id_counter,
            'gene_to_gene_id': serialized_gene_to_gene_id,
            'node_counter': self.node_counter,
            'conn_split_history': serialized_conn_split_history
        }
