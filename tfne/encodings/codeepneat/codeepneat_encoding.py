from .codeepneat_genome import CoDeepNEATGenome
from .codeepneat_optimizer_factory import OptimizerFactory
from .codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from .modules.codeepneat_module_base import CoDeepNEATModuleBase
from .modules import CoDeepNEATModuleDenseDropout, CoDeepNEATModuleConv2DMaxPool2DDropout
from ..base_encoding import BaseEncoding

# Association dict of the module string name to its implementation class
MODULES = {
    'DenseDropout': CoDeepNEATModuleDenseDropout,
    'Conv2DMaxPool2DDropout': CoDeepNEATModuleConv2DMaxPool2DDropout
}


class CoDeepNEATEncoding(BaseEncoding):
    """"""

    def __init__(self, dtype):
        """"""
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

    def create_initial_module(self, mod_type, config_params) -> (int, CoDeepNEATModuleBase):
        """"""
        # Determine module ID and set the parent mutation to 'init' notification
        module_id = self.get_next_module_id()
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        # Create a dict setting all module parameters to None, only because those parameters are required arguments,
        # though let the module self initialize
        module_parameters = dict()
        for section in config_params.keys():
            module_parameters[section] = None

        return module_id, MODULES[mod_type](config_params=config_params,
                                            module_id=module_id,
                                            parent_mutation=parent_mutation,
                                            self_initialize=True,
                                            **module_parameters)

    def get_next_module_id(self) -> int:
        """"""
        self.mod_id_counter += 1
        return self.mod_id_counter

    def create_blueprint(self,
                         blueprint_graph,
                         optimizer_factory,
                         parent_mutation) -> (int, CoDeepNEATBlueprint):
        """"""
        self.bp_id_counter += 1
        return self.bp_id_counter, CoDeepNEATBlueprint(blueprint_id=self.bp_id_counter,
                                                       parent_mutation=parent_mutation,
                                                       blueprint_graph=blueprint_graph,
                                                       optimizer_factory=optimizer_factory)

    def create_blueprint_node(self, node, species) -> (int, CoDeepNEATBlueprintNode):
        """"""
        gene_key = (node,)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintNode(bp_gene_id, node, species)

    def create_blueprint_conn(self, conn_start, conn_end) -> (int, CoDeepNEATBlueprintConn):
        """"""
        gene_key = (conn_start, conn_end)
        if gene_key not in self.gene_to_gene_id:
            self.bp_gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.bp_gene_id_counter

        bp_gene_id = self.gene_to_gene_id[gene_key]
        return bp_gene_id, CoDeepNEATBlueprintConn(bp_gene_id, conn_start, conn_end)

    def get_node_for_split(self, conn_start, conn_end) -> int:
        """"""
        conn_key = (conn_start, conn_end)
        if conn_key not in self.conn_split_history:
            self.node_counter += 1
            self.conn_split_history[conn_key] = self.node_counter

        return self.conn_split_history[conn_key]

    def create_genome(self,
                      blueprint,
                      bp_assigned_modules,
                      output_layers,
                      input_shape,
                      generation) -> (int, CoDeepNEATGenome):
        """"""

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
