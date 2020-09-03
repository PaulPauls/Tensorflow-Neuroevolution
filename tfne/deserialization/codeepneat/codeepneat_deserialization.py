import ast

from tfne.encodings.codeepneat import CoDeepNEATGenome
from tfne.encodings.codeepneat import CoDeepNEATBlueprint
from tfne.encodings.codeepneat.codeepneat_encoding import CoDeepNEATEncoding
from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from tfne.encodings.codeepneat.codeepneat_optimizer_factory import OptimizerFactory
from tfne.encodings.codeepneat.modules import CoDeepNEATModuleBase
from tfne.encodings.codeepneat.modules import MODULES
from tfne.populations.codeepneat.codeepneat_population import CoDeepNEATPopulation


def deserialize_codeepneat_population(serialized_population, dtype, module_config_params=None) -> CoDeepNEATPopulation:
    """
    Deserializes a complete serialized CoDeepNEAT population and returns the CoDeepNEAT population instance
    @param serialized_population: dict serialized CoDeepNEAT population
    @param dtype: string of the TF datatype the CoDeepNEAT population should be deserialized to
    @param module_config_params: dict of module config parameters specyifing the parameter range for all deserialized
                                 modules
    @return: instance of a deserialized CoDeepNEAT population
    """
    # Deserialize all saved population internal evolution information except for the modules, blueprints and best
    # genome, as they have to be deserialized seperately. Save all in the initial state dict of the CoDeepNEAT pop.
    initial_state = dict()

    initial_state['generation_counter'] = serialized_population['generation_counter']
    initial_state['mod_species'] = {int(k): v for k, v in serialized_population['mod_species'].items()}
    initial_state['mod_species_repr'] = {int(k): v for k, v in serialized_population['mod_species_repr'].items()}
    initial_state['mod_species_fitness_history'] = {int(k1): {int(k2): v2 for k2, v2 in v1.items()}
                                                    for k1, v1
                                                    in serialized_population['mod_species_fitness_history'].items()}
    initial_state['mod_species_counter'] = serialized_population['mod_species_counter']
    initial_state['bp_species'] = {int(k): v for k, v in serialized_population['bp_species'].items()}
    initial_state['bp_species_repr'] = {int(k): v for k, v in serialized_population['bp_species_repr'].items()}
    initial_state['bp_species_fitness_history'] = {int(k1): {int(k2): v2 for k2, v2 in v1.items()}
                                                   for k1, v1
                                                   in serialized_population['bp_species_fitness_history'].items()}
    initial_state['bp_species_counter'] = serialized_population['bp_species_counter']
    initial_state['best_fitness'] = serialized_population['best_fitness']

    # Deserialize modules
    initial_state['modules'] = dict()
    for mod_id, mod_params in serialized_population['modules'].items():
        initial_state['modules'][int(mod_id)] = deserialize_codeepneat_module(mod_params, dtype, module_config_params)

    # Deserialize blueprints
    initial_state['blueprints'] = dict()
    for bp_id, bp_params in serialized_population['blueprints'].items():
        initial_state['blueprints'][int(bp_id)] = deserialize_codeepneat_blueprint(bp_params)

    # Deserialize best genome
    initial_state['best_genome'] = deserialize_codeepneat_genome(serialized_population['best_genome'],
                                                                 module_config_params)

    return CoDeepNEATPopulation(initial_state=initial_state)


def deserialize_codeepneat_encoding(serialized_encoding, dtype) -> CoDeepNEATEncoding:
    """
    Deserialize a serialized CoDeepNEAT encoding and return a specific CoDeepNEAT instance.
    @param serialized_encoding: dict serialized CoDeepNEAT encoding
    @param dtype: string of the TF datatype the deserialized CoDeepNEAT encoding should be initialized with
    @return: instance of the deserialized CoDeepNEAT encoding
    """
    # Deserialize all saved population internal evolution information, including the gene_to_gene_id associations and
    # connection split history. Save all in the initial state dict of the CoDeepNEAT encoding.
    inital_state = dict()

    # Convert keys of serialized gene_to_gene_id and conn_split_history dicts back to tuples
    inital_state['gene_to_gene_id'] = dict()
    for key, value in serialized_encoding['gene_to_gene_id'].items():
        deserialized_key = ast.literal_eval(key)
        inital_state['gene_to_gene_id'][deserialized_key] = value
    inital_state['conn_split_history'] = dict()
    for key, value in serialized_encoding['conn_split_history'].items():
        deserialized_key = ast.literal_eval(key)
        inital_state['conn_split_history'][deserialized_key] = value

    # Deserialize rest of encoding state
    inital_state['genome_id_counter'] = serialized_encoding['genome_id_counter']
    inital_state['mod_id_counter'] = serialized_encoding['mod_id_counter']
    inital_state['bp_id_counter'] = serialized_encoding['bp_id_counter']
    inital_state['bp_gene_id_counter'] = serialized_encoding['bp_gene_id_counter']
    inital_state['node_counter'] = serialized_encoding['node_counter']

    return CoDeepNEATEncoding(dtype=dtype, initial_state=inital_state)


def deserialize_codeepneat_genome(serialized_genome, module_config_params=None) -> CoDeepNEATGenome:
    """
    Deserializes a serialized CoDeepNEAT genome genotype and returns a specific CoDeepNEAT genome instance.
    @param serialized_genome: dict serialized CoDeepNEAT genome genotype
    @param module_config_params: dict of module config parameters specyifing the parameter range for all deserialized
                                 modules
    @return: instance of the deserialized CoDeepNEAT genome
    """
    # Deserialize underlying blueprint of genome
    blueprint = deserialize_codeepneat_blueprint(serialized_genome['blueprint'])

    # Deserialize bp_assigned_mods
    bp_assigned_mods = dict()
    for spec, assigned_mod in serialized_genome['bp_assigned_modules'].items():
        bp_assigned_mods[int(spec)] = deserialize_codeepneat_module(assigned_mod,
                                                                    serialized_genome['dtype'],
                                                                    module_config_params)

    # Create genome from deserialized genome parameters and return it
    deserialized_genome = CoDeepNEATGenome(genome_id=serialized_genome['genome_id'],
                                           blueprint=blueprint,
                                           bp_assigned_modules=bp_assigned_mods,
                                           output_layers=serialized_genome['output_layers'],
                                           input_shape=tuple(serialized_genome['input_shape']),
                                           dtype=serialized_genome['dtype'],
                                           origin_generation=serialized_genome['origin_generation'])
    deserialized_genome.set_fitness(serialized_genome['fitness'])
    return deserialized_genome


def deserialize_codeepneat_module(mod_params, dtype, module_config_params=None) -> CoDeepNEATModuleBase:
    """
    Deserializes a serialized CoDeepNEAT module and returns a specific CoDeepNEAT module instance
    @param mod_params: dict serialized parameters of the CoDeepNEAT module
    @param dtype: string of the TF datatype the deserialized CoDeepNEAT module should be initialized with
    @param module_config_params: dict of module config parameters specyifing the parameter range for all deserialized
                                 modules
    @return: instance of the deserialized CoDeepNEAT module
    """
    mod_type = mod_params['module_type']
    del mod_params['module_type']
    # If module is deserialized only for the purpose of inspection/visualization/layer creation, no module config params
    # are required, as those are only needed for mutation/crossover. Therefore, if no module config params are supplied
    # create module accordingly
    if module_config_params is None:
        return MODULES[mod_type](config_params=None,
                                 dtype=dtype,
                                 **mod_params)
    else:
        return MODULES[mod_type](config_params=module_config_params[mod_type],
                                 dtype=dtype,
                                 **mod_params)


def deserialize_codeepneat_blueprint(bp_params) -> CoDeepNEATBlueprint:
    """
    Deserializes a serialized CoDeepNEAT blueprint and returns a specific CoDeepNEAT blueprint instance
    @param bp_params: dict serialized parameters of the CoDeepNEAT blueprint
    @return: instance of the deserialized CoDeepNEAT blueprint
    """
    # Deserialize Blueprint graph
    bp_graph = dict()
    for gene_id, gene_params in bp_params['blueprint_graph'].items():
        if 'node' in gene_params:
            bp_graph[int(gene_id)] = CoDeepNEATBlueprintNode(int(gene_id), gene_params['node'], gene_params['species'])
        else:
            bp_graph[int(gene_id)] = CoDeepNEATBlueprintConn(int(gene_id),
                                                             gene_params['conn_start'],
                                                             gene_params['conn_end'],
                                                             gene_params['enabled'])
    # Recreate optimizer factory
    optimizer_factory = OptimizerFactory(bp_params['optimizer_factory'])

    # Recreate deserialized Blueprint
    return CoDeepNEATBlueprint(bp_params['blueprint_id'],
                               bp_params['parent_mutation'],
                               bp_graph,
                               optimizer_factory)
