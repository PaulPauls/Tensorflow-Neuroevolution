import sys
import json
import math
import random
import statistics

import numpy as np
from absl import logging

import tfne
from ..base_algorithm import BaseNeuroevolutionAlgorithm
from ...encodings.codeepneat.codeepneat_genome import CoDeepNEATGenome
from ...encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprint
from ...encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn
from ...encodings.codeepneat.modules.codeepneat_module_base import CoDeepNEATModuleBase
from ...helper_functions import read_option_from_config, round_with_step


class CoDeepNEAT(BaseNeuroevolutionAlgorithm):
    """"""

    def __init__(self, config, environment, initial_population_file_path=None):
        """"""
        # Read and process the supplied config and register the optionally supplied initial population
        self._process_config(config)
        self.initial_population_file_path = initial_population_file_path

        # Register the supplied environment class and declare the container for the initialized evaluation environments
        # as well as the environment parameters (input and output dimension/shape) to which the created neural networks
        # have to adhere to and which will be set when initializing the environments.
        self.environment = environment
        self.envs = list()
        self.input_shape = None
        self.input_dim = None
        self.output_shape = None
        self.output_dim = None

        # Initialize and register the associated CoDeepNEAT encoding
        self.encoding = tfne.encodings.CoDeepNEATEncoding(dtype=self.dtype)

        # Declare internal variables of the population
        self.generation_counter = None
        self.best_genome = None
        self.best_fitness = None

        # Declare and initialize internal variables concerning the module population of the CoDeepNEAT algorithm
        self.modules = dict()
        self.mod_species = dict()
        self.mod_species_type = dict()
        self.mod_species_counter = 0

        # Declare and initialize internal variables concerning the blueprint population of the CoDeepNEAT algorithm
        self.blueprints = dict()
        self.bp_species = dict()
        self.bp_species_counter = 0

    def _process_config(self, config):
        """"""
        # Read and process the general config values for CoDeepNEAT
        self.dtype = read_option_from_config(config, 'GENERAL', 'dtype')
        self.bp_pop_size = read_option_from_config(config, 'GENERAL', 'bp_pop_size')
        self.mod_pop_size = read_option_from_config(config, 'GENERAL', 'mod_pop_size')
        self.genomes_per_bp = read_option_from_config(config, 'GENERAL', 'genomes_per_bp')
        self.eval_epochs = read_option_from_config(config, 'GENERAL', 'eval_epochs')
        self.eval_batch_size = read_option_from_config(config, 'GENERAL', 'eval_batch_size')

        # Read and process the config values that concern the genome creation for CoDeepNEAT
        self.available_modules = read_option_from_config(config, 'GENOME', 'available_modules')
        self.available_optimizers = read_option_from_config(config, 'GENOME', 'available_optimizers')
        self.preprocessing = read_option_from_config(config, 'GENOME', 'preprocessing')
        self.output_layers = read_option_from_config(config, 'GENOME', 'output_layers')

        # Adjust output_layers config to include the configured datatype
        for out_layer in self.output_layers:
            out_layer['config']['dtype'] = self.dtype

        # Read and process the config values that concern the module speciation for CoDeepNEAT
        self.mod_spec_type = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_type')
        if self.mod_spec_type == 'basic':
            self.mod_spec_min_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_min_size')
            self.mod_spec_max_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_max_size')
            self.mod_spec_elitism = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_elitism')
            self.mod_spec_reprod_thres = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_reprod_thres')
        elif self.mod_spec_type == 'param-distance-fixed':
            self.mod_spec_distance = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_distance')
            self.mod_spec_min_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_min_size')
            self.mod_spec_max_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_max_size')
            self.mod_spec_elitism = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_elitism')
            self.mod_spec_reprod_thres = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_reprod_thres')
            self.mod_spec_max_stagnation = read_option_from_config(config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_max_stagnation')
            self.mod_spec_reinit_extinct = read_option_from_config(config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_reinit_extinct')
        elif self.mod_spec_type == 'param-distance-dynamic':
            self.mod_spec_species_count = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_species_count')
            self.mod_spec_distance = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_distance')
            self.mod_spec_min_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_min_size')
            self.mod_spec_max_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_max_size')
            self.mod_spec_elitism = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_elitism')
            self.mod_spec_reprod_thres = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_reprod_thres')
            self.mod_spec_max_stagnation = read_option_from_config(config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_max_stagnation')
            self.mod_spec_reinit_extinct = read_option_from_config(config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_reinit_extinct')
        else:
            raise NotImplementedError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of modules for CoDeepNEAT
        self.mod_max_mutation = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_max_mutation')
        self.mod_mutation_prob = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_mutation_prob')
        self.mod_crossover_prob = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_crossover_prob')

        # Read and process the config values that concern the blueprint speciation for CoDeepNEAT
        self.bp_spec_type = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_type')
        if self.bp_spec_type == 'basic':
            self.bp_spec_elitism = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_elitism')
            self.bp_spec_reprod_thres = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
        elif self.bp_spec_type == 'gene-overlap-fixed':
            self.bp_spec_distance = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_distance')
            self.bp_spec_min_size = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_min_size')
            self.bp_spec_max_size = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_max_size')
            self.bp_spec_elitism = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_elitism')
            self.bp_spec_reprod_thres = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
            self.bp_spec_max_stagnation = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_max_stagnation')
            self.bp_spec_reinit_extinct = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reinit_extinct')
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            self.bp_spec_species_count = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_species_count')
            self.bp_spec_distance = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_distance')
            self.bp_spec_min_size = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_min_size')
            self.bp_spec_max_size = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_max_size')
            self.bp_spec_elitism = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_elitism')
            self.bp_spec_reprod_thres = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
            self.bp_spec_max_stagnation = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_max_stagnation')
            self.bp_spec_reinit_extinct = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reinit_extinct')
        else:
            raise NotImplementedError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of blueprints for CoDeepNEAT
        self.bp_max_mutation = read_option_from_config(config, 'BP_EVOLUTION', 'bp_max_mutation')
        self.bp_mutation_add_conn_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_conn_prob')
        self.bp_mutation_add_node_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_node_prob')
        self.bp_mutation_rem_conn_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_rem_conn_prob')
        self.bp_mutation_rem_node_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_rem_node_prob')
        self.bp_mutation_node_spec_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_node_spec_prob')
        self.bp_mutation_optimizer_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_optimizer_prob')
        self.bp_crossover_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_crossover_prob')

        # Read and process the config values that concern the parameter range of the modules for CoDeepNEAT
        self.available_mod_params = dict()
        for available_mod in self.available_modules:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'MODULE_' + available_mod.upper()
            if not config.has_section(config_section_str):
                raise RuntimeError(f"Module '{available_mod}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            mod_section_params = dict()
            for mod_param in config.options(config_section_str):
                mod_section_params[mod_param] = read_option_from_config(config, config_section_str, mod_param)

            # Assign that dict of all available parameters for the module to the instance variable
            self.available_mod_params[available_mod] = mod_section_params

        # Read and process the config values that concern the parameter range of the optimizers for CoDeepNEAT
        self.available_opt_params = dict()
        for available_opt in self.available_optimizers:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'OPTIMIZER_' + available_opt.upper()
            if not config.has_section(config_section_str):
                raise RuntimeError(f"Optimizer '{available_opt}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            opt_section_params = dict()
            for opt_param in config.options(config_section_str):
                opt_section_params[opt_param] = read_option_from_config(config, config_section_str, opt_param)

            # Assign that dict of all available parameters for the optimizers to the instance variable
            self.available_opt_params[available_opt] = opt_section_params

        # Perform some basic sanity checks of the configuration
        if self.mod_spec_type == 'basic':
            assert self.mod_spec_min_size * len(self.available_modules) <= self.mod_pop_size
            assert self.mod_spec_max_size * len(self.available_modules) >= self.mod_pop_size
        elif self.mod_spec_type == 'param-distance-fixed':
            assert self.mod_spec_min_size * len(self.available_modules) <= self.mod_pop_size
        elif self.mod_spec_type == 'param-distance-dynamic':
            assert self.mod_spec_min_size * len(self.available_modules) <= self.mod_pop_size
            assert self.mod_spec_min_size \
                   <= int(self.mod_pop_size / self.mod_spec_species_count) \
                   <= self.mod_spec_max_size
        if self.bp_spec_type == 'gene-overlap-fixed':
            assert self.bp_spec_min_size <= self.bp_pop_size
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            assert self.bp_spec_min_size <= self.bp_pop_size
            assert self.bp_spec_min_size <= int(self.bp_pop_size / self.bp_spec_species_count) <= self.bp_spec_max_size
        assert round(self.mod_mutation_prob + self.mod_crossover_prob, 4) == 1.0
        assert round(self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob + self.bp_mutation_rem_conn_prob
                     + self.bp_mutation_rem_node_prob + self.bp_mutation_node_spec_prob + self.bp_crossover_prob
                     + self.bp_mutation_optimizer_prob, 4) == 1.0

    def initialize_environments(self, num_cpus, num_gpus, verbosity):
        """"""
        # TODO Implement algorithm parallelisation and initialization of multiple environments
        # Initialize only one instance as implementation currently only supports single instance evaluation
        for _ in range(1):
            initialized_env = self.environment(weight_training=True,
                                               verbosity=verbosity,
                                               epochs=self.eval_epochs,
                                               batch_size=self.eval_batch_size)
            self.envs.append(initialized_env)

        # Determine required input and output dimensions and shape
        self.input_shape = self.envs[0].get_input_shape()
        self.input_dim = len(self.input_shape)
        self.output_shape = self.envs[0].get_output_shape()
        self.output_dim = len(self.output_shape)

    def initialize_population(self):
        """"""
        if self.initial_population_file_path is None:
            print("Initializing a new population of {} blueprints and {} modules..."
                  .format(self.bp_pop_size, self.mod_pop_size))

            # Set internal variables of the population to the initialization of a new population
            self.generation_counter = 0
            self.best_fitness = 0

            #### Initialize Module Population ####
            # Initialize module population with a basic speciation scheme, even when another speciation type is supplied
            # as config, only speciating modules according to their module type. Each module species (and therefore
            # module type) is initiated with the same amount of modules (or close to the same amount if module pop size
            # not evenly divisble). Parameters of all initial modules chosen as per module implementation (though
            # usually uniformly random)

            # Set initial species counter of basic speciation, initialize module species list and map each species to
            # its type
            for mod_type in self.available_modules:
                self.mod_species_counter += 1
                self.mod_species[self.mod_species_counter] = list()
                self.mod_species_type[self.mod_species_counter] = mod_type

            for i in range(self.mod_pop_size):
                # Decide on for which species a new module is added (uniformly distributed)
                chosen_species = (i % self.mod_species_counter) + 1

                # Determine type and the associated config parameters of chosen species and initialize a module with it
                mod_type = self.mod_species_type[chosen_species]
                mod_config_params = self.available_mod_params[mod_type]
                module_id, module = self.encoding.create_initial_module(mod_type=mod_type,
                                                                        config_params=mod_config_params)

                # Append newly created initial module to module container and to according species
                self.modules[module_id] = module
                self.mod_species[chosen_species].append(module_id)

            #### Initialize Blueprint Population ####
            # Initialize blueprint population with a minimal blueprint graph, only consisting of an input node (with
            # None species or the 'input' species respectively) and a single output node, having a randomly assigned
            # species. All hyperparameters of the blueprint are uniform randomly chosen. All blueprints are not
            # speciated in the beginning and are assigned to species 1.

            # Initialize blueprint species list and create tuple of the possible species the output node can take on
            self.bp_species[1] = list()
            available_mod_species = tuple(self.mod_species.keys())

            for _ in range(self.bp_pop_size):
                # Determine the module species of the initial (and only) node
                initial_node_species = random.choice(available_mod_species)

                # Initialize a new blueprint with minimal graph only using initial node species
                blueprint_id, blueprint = self._create_initial_blueprint(initial_node_species)

                # Append newly create blueprint to blueprint container and to only initial blueprint species
                self.blueprints[blueprint_id] = blueprint
                self.bp_species[1].append(blueprint_id)
        else:
            raise NotImplementedError("Initializing population with pre-evolved initial population not yet implemented")

    def _create_initial_blueprint(self, initial_node_species) -> (int, CoDeepNEATBlueprint):
        """"""
        # Create the dict that keeps track of the way a blueprint has been mutated or created
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        # Create a minimal blueprint graph with node 1 being the input node (having no species) and node 2 being the
        # random initial node species
        blueprint_graph = dict()
        gene_id, gene = self.encoding.create_blueprint_node(node=1, species=None)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.encoding.create_blueprint_node(node=2, species=initial_node_species)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.encoding.create_blueprint_conn(conn_start=1, conn_end=2)
        blueprint_graph[gene_id] = gene

        # Randomly choose an optimizer from the available optimizers and create the parameter config dict of it
        chosen_optimizer = random.choice(self.available_optimizers)
        available_optimizer_params = self.available_opt_params[chosen_optimizer]

        # Declare container collecting the specific parameters of the optimizer to be created, setting the just chosen
        # optimizer class
        chosen_optimizer_params = {'class_name': chosen_optimizer, 'config': dict()}

        # Traverse each possible parameter option and determine a uniformly random value depending on if its a
        # categorical, sortable or boolean value
        for opt_param, opt_param_val_range in available_optimizer_params.items():
            # If the optimizer parameter is a categorical value choose randomly from the list
            if isinstance(opt_param_val_range, list):
                chosen_optimizer_params['config'][opt_param] = random.choice(opt_param_val_range)
            # If the optimizer parameter is sortable, create a random value between the min and max values adhering
            # to the configured step
            elif isinstance(opt_param_val_range, dict):
                if isinstance(opt_param_val_range['min'], int) and isinstance(opt_param_val_range['max'], int) \
                        and isinstance(opt_param_val_range['step'], int):
                    opt_param_random = random.randint(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                elif isinstance(opt_param_val_range['min'], float) and isinstance(opt_param_val_range['max'], float) \
                        and isinstance(opt_param_val_range['step'], float):
                    opt_param_random = random.uniform(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round(round_with_step(opt_param_random,
                                                             opt_param_val_range['min'],
                                                             opt_param_val_range['max'],
                                                             opt_param_val_range['step']), 4)
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer "
                                              f"section is of type dict though the dict values are not of type int or "
                                              f"float")
                chosen_optimizer_params['config'][opt_param] = chosen_opt_param
            # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
            # parameter being set to True
            elif isinstance(opt_param_val_range, float):
                chosen_optimizer_params['config'][opt_param] = random.random() < opt_param_val_range
            else:
                raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer section "
                                          f"is not one of the valid types of list, dict or float")

        # Create new optimizer through encoding
        optimizer_factory = self.encoding.create_optimizer_factory(optimizer_parameters=chosen_optimizer_params)

        # Create just defined initial blueprint through encoding
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def evaluate_population(self) -> (int, int):
        """"""
        # Create container collecting the fitness of the genomes that involve specific modules. Calculate the average
        # fitness of the genomes in which a module is involved in later and assign it as the module's fitness
        mod_fitnesses_in_genomes = dict()

        # Initialize Progress counter variables for evaluate population progress bar. Print notice of evaluation start
        genome_pop_size = self.bp_pop_size * self.genomes_per_bp
        genome_eval_counter = 0
        genome_eval_counter_div = round(genome_pop_size / 40.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(genome_pop_size, self.generation_counter))
        print_str = "\r[{:40}] {}/{} Genomes".format("", genome_eval_counter, genome_pop_size)
        sys.stdout.write(print_str)
        sys.stdout.flush()

        for blueprint in self.blueprints.values():
            bp_module_species = blueprint.get_species()

            # Create container collecting the fitness of the genomes that involve that specific blueprint.
            bp_fitnesses_in_genomes = list()

            for _ in range(self.genomes_per_bp):
                # Assemble genome by first uniform randomly choosing a specific module from the species that the
                # blueprint nodes are referring to.
                bp_assigned_modules = dict()
                for i in bp_module_species:
                    chosen_module_id = random.choice(self.mod_species[i])
                    bp_assigned_modules[i] = self.modules[chosen_module_id]

                try:
                    # Create genome, using the specific blueprint, a dict of modules for each species, the configured
                    # output layers and input shape as well as the current generation
                    genome_id, genome = self.encoding.create_genome(blueprint,
                                                                    bp_assigned_modules,
                                                                    self.output_layers,
                                                                    self.input_shape,
                                                                    self.generation_counter)
                except ValueError:
                    # Catching build value error, occuring when the supplied layers and parameters do not result in a
                    # valid TF model. See warning string.
                    bp_id = blueprint.get_id()
                    mod_spec_to_id = dict()
                    for spec, mod in bp_assigned_modules.items():
                        mod_spec_to_id[spec] = mod.get_id()
                    logging.warning(f"CoDeepNEAT tried combining the Blueprint ID {bp_id} with the module assignment "
                                    f"{mod_spec_to_id}, resulting in an invalid neural network model. Setting genome "
                                    f"fitness to 0.")

                    # Setting genome id and genome to None as referenced later. Setting genome fitness to 0 to
                    # discourage continued use of the blueprint and modules resulting in this faulty model.
                    genome_id, genome = None, None
                    genome_fitness = 0

                if genome is not None:
                    # Now evaluate genome on registered environment and set its fitness
                    # NOTE: As CoDeepNEAT implementation currently only supports 1 eval instance, automatically choose
                    # that instance from the environment list
                    genome_fitness = self.envs[0].eval_genome_fitness(genome)
                    genome.set_fitness(genome_fitness)

                # Print population evaluation progress bar
                genome_eval_counter += 1
                progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
                print_str = "\r[{:40}] {}/{} Genomes | Genome ID {} achieved fitness of {}".format("=" * progress_mult,
                                                                                                   genome_eval_counter,
                                                                                                   genome_pop_size,
                                                                                                   genome_id,
                                                                                                   genome_fitness)
                sys.stdout.write(print_str)
                sys.stdout.flush()

                # Assign the genome fitness to the blueprint and all modules used for the creation of the genome
                bp_fitnesses_in_genomes.append(genome_fitness)
                for assigned_module in bp_assigned_modules.values():
                    module_id = assigned_module.get_id()
                    if module_id in mod_fitnesses_in_genomes:
                        mod_fitnesses_in_genomes[module_id].append(genome_fitness)
                    else:
                        mod_fitnesses_in_genomes[module_id] = [genome_fitness]

                # Register genome as new best if it exhibits better fitness than the previous best
                if self.best_fitness is None or genome_fitness > self.best_fitness:
                    self.best_genome = genome
                    self.best_fitness = genome_fitness

            # Average out collected fitness of genomes the blueprint was invovled in. Then assign that average fitness
            # to the blueprint
            bp_fitnesses_in_genomes_avg = round(statistics.mean(bp_fitnesses_in_genomes), 4)
            blueprint.set_fitness(bp_fitnesses_in_genomes_avg)

        # Average out collected fitness of genomes each module was invovled in. Then assign that average fitness to the
        # module
        for mod_id, mod_fitness_list in mod_fitnesses_in_genomes.items():
            mod_genome_fitness_avg = round(statistics.mean(mod_fitness_list), 4)
            self.modules[mod_id].set_fitness(mod_genome_fitness_avg)

        return self.generation_counter, self.best_fitness

    def summarize_population(self):
        """"""
        # Calculate average fitnesses of each module species and total module pop. Determine best module of each species
        mod_species_avg_fitness = dict()
        mod_species_best_id = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_total_fitness = 0
            for mod_id in spec_mod_ids:
                mod_fitness = self.modules[mod_id].get_fitness()
                spec_total_fitness += mod_fitness
                if spec_id not in mod_species_best_id or mod_fitness > mod_species_best_id[spec_id][1]:
                    mod_species_best_id[spec_id] = (mod_id, mod_fitness)
            mod_species_avg_fitness[spec_id] = round(spec_total_fitness / len(spec_mod_ids), 4)
        modules_avg_fitness = round(sum(mod_species_avg_fitness.values()) / len(mod_species_avg_fitness), 4)

        # Calculate average fitnesses of each bp species and total bp pop. Determine best bp of each species
        bp_species_avg_fitness = dict()
        bp_species_best_id = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            spec_total_fitness = 0
            for bp_id in spec_bp_ids:
                bp_fitness = self.blueprints[bp_id].get_fitness()
                spec_total_fitness += bp_fitness
                if spec_id not in bp_species_best_id or bp_fitness > bp_species_best_id[spec_id][1]:
                    bp_species_best_id[spec_id] = (bp_id, bp_fitness)
            bp_species_avg_fitness[spec_id] = round(spec_total_fitness / len(spec_bp_ids), 4)
        blueprints_avg_fitness = round(sum(bp_species_avg_fitness.values()) / len(bp_species_avg_fitness), 4)

        # Print summary header
        print("\n\n\n\033[1m{}  Population Summary  {}\n\n"
              "Generation: {:>4}  ||  Best Genome Fitness: {:>8}  ||  Average Blueprint Fitness: {:>8}  ||  "
              "Average Module Fitness: {:>8}\033[0m\n"
              "Best Genome: {}\n".format('#' * 60,
                                         '#' * 60,
                                         self.generation_counter,
                                         self.best_fitness,
                                         blueprints_avg_fitness,
                                         modules_avg_fitness,
                                         self.best_genome))

        # Print summary of blueprint species
        print("\033[1mBP Species                || BP Species Avg Fitness                || BP Species Size\n"
              "Best BP of Species\033[0m")
        for spec_id, spec_bp_avg_fitness in bp_species_avg_fitness.items():
            print("{:>6}                    || {:>8}                              || {:>8}\n{}"
                  .format(spec_id,
                          spec_bp_avg_fitness,
                          len(self.bp_species[spec_id]),
                          self.blueprints[bp_species_best_id[spec_id][0]]))

        # Print summary of module species
        print("\n\033[1mModule Species            || Module Species Avg Fitness            || "
              "Module Species Size\nBest Module of Species\033[0m")
        for spec_id, spec_mod_avg_fitness in mod_species_avg_fitness.items():
            print("{:>6}                    || {:>8}                              || {:>8}\n{}"
                  .format(spec_id,
                          spec_mod_avg_fitness,
                          len(self.mod_species[spec_id]),
                          self.modules[mod_species_best_id[spec_id][0]]))

        # Print summary footer
        print("\n\033[1m" + '#' * 142 + "\033[0m\n")

    def evolve_population(self) -> bool:
        """"""
        #### Speciate Modules ####
        if self.mod_spec_type == 'basic':
            new_modules, new_mod_species, new_mod_species_size = self._speciate_modules_basic()
        elif self.mod_spec_type == 'param-distance-fixed':
            new_modules, new_mod_species, new_mod_species_size = self._speciate_modules_param_distance_fixed()
        elif self.mod_spec_type == 'param-distance-dynamic':
            new_modules, new_mod_species, new_mod_species_size = self._speciate_modules_param_distance_dynamic()
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        #### Speciate Blueprints ####
        if self.bp_spec_type == 'basic':
            new_blueprints, new_bp_species, new_bp_species_size = self._speciate_blueprints_basic()
        elif self.bp_spec_type == 'gene-overlap-fixed':
            new_blueprints, new_bp_species, new_bp_species_size = self._speciate_blueprints_gene_overlap_fixed()
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            new_blueprints, new_bp_species, new_bp_species_size = self._speciate_blueprints_gene_overlap_dynamic()
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        #### Evolve Modules ####
        # Traverse through the new module species and add new modules until calculated dedicated spec size is reached
        for spec_id, carried_over_mod_ids in new_mod_species.items():
            # Determine amount of offspring and create according amount of new modules
            for _ in range(new_mod_species_size[spec_id] - len(carried_over_mod_ids)):
                # Choose randomly between mutation or crossover of module
                if random.random() < self.mod_mutation_prob:
                    ## Create new module through mutation ##
                    # Get a new module ID from the encoding, randomly determine the maximum degree of mutation and the
                    # parent module from the non removed modules of the current species. Then let the internal mutation
                    # function create a new module
                    mod_offspring_id = self.encoding.get_next_module_id()
                    max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                    parent_module = self.modules[random.choice(self.mod_species[spec_id])]

                    new_mod_id, new_mod = parent_module.create_mutation(mod_offspring_id,
                                                                        max_degree_of_mutation)

                else:  # random.random() < self.mod_mutation_prob + self.mod_crossover_prob
                    ## Create new module through crossover ##
                    # Determine if species has at least 2 modules as required for crossover
                    if len(self.mod_species[spec_id]) >= 2:
                        # Determine the 2 parent modules used for crossover
                        parent_module_1_id, parent_module_2_id = random.sample(self.mod_species[spec_id], k=2)
                        parent_module_1 = self.modules[parent_module_1_id]
                        parent_module_2 = self.modules[parent_module_2_id]

                        # Get a new module ID from encoding, randomly determine the maximum degree of mutation
                        mod_offspring_id = self.encoding.get_next_module_id()
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)

                        # Determine the fitter parent module and let its internal crossover function create offspring
                        if parent_module_1.get_fitness() >= parent_module_2.get_fitness():
                            new_mod_id, new_mod = parent_module_1.create_crossover(mod_offspring_id,
                                                                                   parent_module_2,
                                                                                   max_degree_of_mutation)
                        else:
                            new_mod_id, new_mod = parent_module_2.create_crossover(mod_offspring_id,
                                                                                   parent_module_1,
                                                                                   max_degree_of_mutation)

                    else:
                        # As species does not have enough modules for crossover, perform a mutation on the remaining
                        # module
                        mod_offspring_id = self.encoding.get_next_module_id()
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                        parent_module = self.modules[random.choice(self.mod_species[spec_id])]

                        new_mod_id, new_mod = parent_module.create_mutation(mod_offspring_id,
                                                                            max_degree_of_mutation)

                # Add newly created module to the module container and its according species
                new_modules[new_mod_id] = new_mod
                new_mod_species[spec_id].append(new_mod_id)

        # As new module container and species dict have now been fully created, overwrite the old ones
        self.modules = new_modules
        self.mod_species = new_mod_species

        #### Evolve Blueprints ####
        # Calculate the brackets for a random float to fall into in order to choose a specific evolutionary method
        bp_mutation_add_node_bracket = self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob
        bp_mutation_rem_conn_bracket = bp_mutation_add_node_bracket + self.bp_mutation_rem_conn_prob
        bp_mutation_rem_node_bracket = bp_mutation_rem_conn_bracket + self.bp_mutation_rem_node_prob
        bp_mutation_node_spec_bracket = bp_mutation_rem_node_bracket + self.bp_mutation_node_spec_prob
        bp_mutation_optimizer_bracket = bp_mutation_node_spec_bracket + self.bp_mutation_optimizer_prob

        # Traverse through the new blueprint species and add new blueprints until calculated dedicated spec size reached
        for spec_id, carried_over_bp_ids in new_bp_species.items():
            # Determine amount of offspring and create according amount of new blueprints
            for _ in range(new_bp_species_size[spec_id] - len(carried_over_bp_ids)):
                # Choose random float vaue determining specific evolutionary method to evolve the chosen blueprint
                random_choice = random.random()
                if random_choice < self.bp_mutation_add_conn_prob:
                    ## Create new blueprint by adding connection ##
                    # Randomly determine the parent blueprint from the current species and the degree of mutation.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_add_node_bracket:
                    ## Create new blueprint by adding node ##
                    # Randomly determine the parent blueprint from the current species and the degree of mutation.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_conn_bracket:
                    ## Create new blueprint by removing connection ##
                    # Randomly determine the parent blueprint from the current species and the degree of mutation.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_node_bracket:
                    ## Create new blueprint by removing node ##
                    # Randomly determine the parent blueprint from the current species and the degree of mutation.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_node_spec_bracket:
                    ## Create new blueprint by mutating species in nodes ##
                    # Randomly determine the parent blueprint from the current species and the degree of mutation.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_node_spec(parent_blueprint,
                                                                                 max_degree_of_mutation)

                elif random_choice < bp_mutation_optimizer_bracket:
                    ## Create new blueprint by mutating the associated optimizer ##
                    # Randomly determine the parent blueprint from the current species.
                    parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                    new_bp_id, new_bp = self._create_mutated_blueprint_optimizer(parent_blueprint)

                else:  # random_choice < bp_crossover_bracket:
                    ## Create new blueprint through crossover ##
                    # Determine if species has at least 2 blueprints as required for crossover
                    if len(self.bp_species[spec_id]) >= 2:
                        # Randomly determine both parents for the blueprint crossover
                        parent_bp_1_id, parent_bp_2_id = random.sample(self.bp_species[spec_id], k=2)
                        parent_bp_1 = self.blueprints[parent_bp_1_id]
                        parent_bp_2 = self.blueprints[parent_bp_2_id]
                        new_bp_id, new_bp = self._create_crossed_over_blueprint(parent_bp_1,
                                                                                parent_bp_2)

                    else:
                        # As species does not have enough blueprints for crossover, perform a simple species
                        # perturbation in the blueprint nodes. Determine uniform randomly the parent blueprint from the
                        # current species and the degree of mutation.
                        parent_blueprint = self.blueprints[random.choice(self.bp_species[spec_id])]
                        max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                        new_bp_id, new_bp = self._create_mutated_blueprint_node_spec(parent_blueprint,
                                                                                     max_degree_of_mutation)

                # Add newly created blueprint to the blueprint container and its according species
                new_blueprints[new_bp_id] = new_bp
                new_bp_species[spec_id].append(new_bp_id)

        # As new blueprint container and species dict have now been fully created, overwrite the old ones
        self.blueprints = new_blueprints
        self.bp_species = new_bp_species

        #### Return ####
        # Adjust generation counter and return False, signalling that the population has not gone extinct
        self.generation_counter += 1
        return False

    def _speciate_modules_basic(self) -> ({int: CoDeepNEATModuleBase}, {int: [int, ...]}, {int: int}):
        """"""
        #### Module Clustering ####
        # As module population is by default speciated by module type is further clustering not necessary
        pass

        #### New Species Size Calculation ####
        # Determine average fitness of each current species as well as the sum of each avg fitness
        mod_species_avg_fitness = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_avg_fitness = statistics.mean([self.modules[mod_id].get_fitness() for mod_id in spec_mod_ids])
            mod_species_avg_fitness[spec_id] = spec_avg_fitness
        total_avg_fitness = sum(mod_species_avg_fitness.values())

        # Calculate the new_mod_species_size depending on the species fitness share of the total fitness.
        new_mod_species_size = dict()
        current_total_size = 0
        for spec_id, spec_avg_fitness in mod_species_avg_fitness.items():
            spec_size = math.floor((spec_avg_fitness / total_avg_fitness) * self.mod_pop_size)

            # If calculated species size violates config specified min/max size correct it
            if spec_size > self.mod_spec_max_size:
                spec_size = self.mod_spec_max_size
            elif spec_size < self.mod_spec_min_size:
                spec_size = self.mod_spec_min_size

            new_mod_species_size[spec_id] = spec_size
            current_total_size += spec_size

        # Flooring / Min / Max species size adjustments likely perturbed the assigned species size in that they don't
        # sum up to the desired module pop size. Decrease or increase new mod species size accordingly.
        while current_total_size < self.mod_pop_size:
            # Increase new mod species size by awarding offspring to species with the currently least assigned offspring
            min_mod_spec_id = min(new_mod_species_size.keys(), key=new_mod_species_size.get)
            new_mod_species_size[min_mod_spec_id] += 1
            current_total_size += 1
        while current_total_size > self.mod_pop_size:
            # Decrease new mod species size by removing offspring from species with currently most assigned offspring
            max_mod_spec_id = max(new_mod_species_size.keys(), key=new_mod_species_size.get)
            new_mod_species_size[max_mod_spec_id] -= 1
            current_total_size -= 1

        #### Module Selection ####
        # Declare new modules container and new module species assignment and carry over x number of best performing
        # modules of each species according to config specified elitism
        new_modules = dict()
        new_mod_species = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            # Sort module ids in species according to their fitness
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.modules[x].get_fitness(), reverse=True)

            # Determine carried over module ids and module ids prevented from reproduction
            spec_mod_ids_to_carry_over = spec_mod_ids_sorted[:self.mod_spec_elitism]
            removal_index_threshold = int(len(spec_mod_ids) * (1 - self.mod_spec_reprod_thres))
            # Correct removal index threshold if reproduction threshold so high that elitism modules will be removed
            if removal_index_threshold + self.mod_spec_elitism < len(spec_mod_ids):
                removal_index_threshold = self.mod_spec_elitism
            spec_mod_ids_to_remove = spec_mod_ids_sorted[removal_index_threshold:]

            # Carry over fittest module ids of species to new container and species assignment
            new_mod_species[spec_id] = list()
            for mod_id_to_carry_over in spec_mod_ids_to_carry_over:
                new_modules[mod_id_to_carry_over] = self.modules[mod_id_to_carry_over]
                new_mod_species[spec_id].append(mod_id_to_carry_over)

            # Delete low performing modules that will not be considered for reproduction from old species assignment
            for mod_id_to_remove in spec_mod_ids_to_remove:
                self.mod_species[spec_id].remove(mod_id_to_remove)

        return new_modules, new_mod_species, new_mod_species_size

    def _speciate_modules_param_distance_fixed(self) -> ({int: CoDeepNEATModuleBase}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _speciate_modules_param_distance_dynamic(self) -> ({int: CoDeepNEATModuleBase}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _speciate_blueprints_basic(self) -> ({int: CoDeepNEATBlueprint}, {int: [int, ...]}, {int: int}):
        """"""
        #### Blueprint Clustering ####
        # Blueprint clustering unnecessary with basic scheme as all blueprints are assigned to species 1
        pass

        #### New Species Size Calculation ####
        # Species size calculation unnecessary as only one species of blueprints will exist containing all bps
        new_bp_species_size = {1: self.bp_pop_size}

        #### Blueprint Selection ####
        # Declare new blueprints container and new blueprint species assignment and carry over x number of best
        # performing blueprints of each species according to config specified elitism
        new_blueprints = dict()
        new_bp_species = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            # Sort blueprint ids in species according to their fitness
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.blueprints[x].get_fitness(), reverse=True)

            # Determine carried over blueprint ids and blueprint ids prevented from reproduction
            spec_bp_ids_to_carry_over = spec_bp_ids_sorted[:self.bp_spec_elitism]
            removal_index_threshold = int(len(spec_bp_ids) * (1 - self.bp_spec_reprod_thres))
            # Correct removal index threshold if reproduction threshold so high that elitism blueprints will be removed
            if removal_index_threshold + self.bp_spec_elitism < len(spec_bp_ids):
                removal_index_threshold = self.bp_spec_elitism
            spec_bp_ids_to_remove = spec_bp_ids_sorted[removal_index_threshold:]

            # Carry over fittest blueprint ids of species to new container and species assignment
            new_bp_species[spec_id] = list()
            for bp_id_to_carry_over in spec_bp_ids_to_carry_over:
                new_blueprints[bp_id_to_carry_over] = self.blueprints[bp_id_to_carry_over]
                new_bp_species[spec_id].append(bp_id_to_carry_over)

            # Delete low performing blueprints that will not be considered for reproduction from old species assignment
            for bp_id_to_remove in spec_bp_ids_to_remove:
                self.bp_species[spec_id].remove(bp_id_to_remove)

        return new_blueprints, new_bp_species, new_bp_species_size

    def _speciate_blueprints_gene_overlap_fixed(self) -> ({int: CoDeepNEATBlueprint}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _speciate_blueprints_gene_overlap_dynamic(self) -> ({int: CoDeepNEATBlueprint}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _create_mutated_blueprint_add_conn(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint and get the pre-analyzed topology of the parent graph
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()
        bp_graph_topology = parent_blueprint.get_graph_topology()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'add_conn',
                           'added_genes': list()}

        # Create collections of all nodes and present connections in the copied blueprint graph
        bp_graph_conns = set()
        bp_graph_nodes = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                bp_graph_nodes.append(gene.node)
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn)
                # Only consider a connection for bp_graph_conns if it is enabled
                bp_graph_conns.add((gene.conn_start, gene.conn_end))

        # Remove end-node (node 2) from this list and shuffle it, as it later serves to randomly pop the start node of
        # the newly created connection
        bp_graph_nodes.remove(2)
        random.shuffle(bp_graph_nodes)

        # Determine specifically how many connections will be added
        number_of_conns_to_add = math.ceil(max_degree_of_mutation * len(bp_graph_conns))

        # Add connections in a loop until predetermined number of connections that are to be added is reached or until
        # the possible starting nodes run out
        added_conns_counter = 0
        while added_conns_counter < number_of_conns_to_add and len(bp_graph_nodes) > 0:
            # Choose random start node from all possible nodes by popping it from a preshuffled list of graph nodes
            start_node = bp_graph_nodes.pop()

            # Determine the list of all possible end nodes that are behind the start node as implementation currently
            # only supports feedforward topologies. Then shuffle the end nodes as they will later be randomly popped
            start_node_level = None
            for i in range(len(bp_graph_topology)):
                if start_node in bp_graph_topology[i]:
                    start_node_level = i
                    break
            possible_end_nodes = list(set().union(*bp_graph_topology[start_node_level + 1:]))
            random.shuffle(possible_end_nodes)

            # Traverse all possible end nodes randomly and create and add a blueprint connection to the offspring
            # blueprint graph if the specific connection tuple is not yet present.
            while len(possible_end_nodes) > 0:
                end_node = possible_end_nodes.pop()
                if (start_node, end_node) not in bp_graph_conns:
                    gene_id, gene = self.encoding.create_blueprint_conn(conn_start=start_node,
                                                                        conn_end=end_node)
                    blueprint_graph[gene_id] = gene
                    parent_mutation['added_genes'].append(gene_id)
                    added_conns_counter += 1

        # Create and return the offspring blueprint with the edited blueprint graph having additional connections as
        # well as the parent duplicated optimizer factory.
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_mutated_blueprint_add_node(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'add_node',
                           'added_genes': list()}

        # Analyze amount of nodes already present in bp graph as well as collect all gene ids of the present connections
        # that can possibly be split
        node_count = 0
        bp_graph_conn_ids = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                node_count += 1
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn)
                # Only consider a connection for bp_graph_conn_ids if it is enabled
                bp_graph_conn_ids.append(gene.gene_id)

        # Determine how many nodes will be added, which connection gene_ids will be split for that and what possible
        # species can be assigned to those new nodes
        number_of_nodes_to_add = math.ceil(max_degree_of_mutation * node_count)
        gene_ids_to_split = random.sample(bp_graph_conn_ids, k=number_of_nodes_to_add)
        available_mod_species = tuple(self.mod_species.keys())

        # Split all chosen connections by setting them to disabled, querying the new node id from the encoding and then
        # creating the new node and the associated 2 connections through the encoding.
        for gene_id_to_split in gene_ids_to_split:
            # Determine start and end node of connection and disable it
            conn_start = blueprint_graph[gene_id_to_split].conn_start
            conn_end = blueprint_graph[gene_id_to_split].conn_end
            blueprint_graph[gene_id_to_split].set_enabled(False)

            # Create a new unique node if connection has not yet been split by any other mutation. Otherwise create the
            # same node. Choose species for new node randomly.
            new_node = self.encoding.get_node_for_split(conn_start, conn_end)
            new_species = random.choice(available_mod_species)

            # Create the node and connections genes for the new node addition and add them to the blueprint graph
            gene_id, gene = self.encoding.create_blueprint_node(node=new_node, species=new_species)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.encoding.create_blueprint_conn(conn_start=conn_start, conn_end=new_node)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.encoding.create_blueprint_conn(conn_start=new_node, conn_end=conn_end)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)

        # Create and return the offspring blueprint with the edited blueprint graph having a new node through a split
        # connection as well as the parent duplicated optimizer factory.
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_mutated_blueprint_rem_conn(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'rem_conn',
                           'removed_genes': list()}

        # Analyze amount of connections already present in bp graph and collect all gene ids whose connection ends in
        # certain nodes, allowing the algorithm to determine which connections can be removed as they are not the sole
        # connection to a remaining node.
        conn_count = 0
        bp_graph_incoming_conn_ids = dict()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintConn) and gene.enabled:
                conn_count += 1
                if gene.conn_end in bp_graph_incoming_conn_ids:
                    bp_graph_incoming_conn_ids[gene.conn_end].append(gene.gene_id)
                else:
                    bp_graph_incoming_conn_ids[gene.conn_end] = [gene.gene_id]

        # Remove all nodes from the 'bp_graph_incoming_conn_ids' dict that have only 1 incoming connection, as this
        # connection is essential and can not be removed without also effectively removing nodes. If a node has more
        # than 1 incoming connection then shuffle those, as they will later be popped.
        bp_graph_incoming_conn_ids_to_remove = list()
        for conn_end, incoming_conn_ids in bp_graph_incoming_conn_ids.items():
            if len(incoming_conn_ids) == 1:
                bp_graph_incoming_conn_ids_to_remove.append(conn_end)
            else:
                random.shuffle(bp_graph_incoming_conn_ids[conn_end])
        for conn_id_to_remove in bp_graph_incoming_conn_ids_to_remove:
            del bp_graph_incoming_conn_ids[conn_id_to_remove]

        # Determine how many conns will be removed based on the total connection count
        number_of_conns_to_rem = math.ceil(max_degree_of_mutation * conn_count)

        # Remove connections in loop until determined number of connections are removed or until no node has 2 incoming
        # connections. Remove connections by randomly choosing a node with more than 1 incoming connections and then
        # removing the associated gene id from the bp graph
        rem_conns_counter = 0
        while rem_conns_counter < number_of_conns_to_rem and len(bp_graph_incoming_conn_ids) > 0:
            rem_conn_end_node = random.choice(tuple(bp_graph_incoming_conn_ids.keys()))
            conn_id_to_remove = bp_graph_incoming_conn_ids[rem_conn_end_node].pop()
            # If node has only 1 incoming connection, remove it from the possible end nodes for future iterations
            if len(bp_graph_incoming_conn_ids[rem_conn_end_node]) == 1:
                del bp_graph_incoming_conn_ids[rem_conn_end_node]

            del blueprint_graph[conn_id_to_remove]
            parent_mutation['removed_genes'].append(conn_id_to_remove)
            rem_conns_counter += 1

        # Create and return the offspring blueprint with the edited blueprint graph having one or multiple connections
        # removed though still having at least 1 connection to each node.
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_mutated_blueprint_rem_node(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'rem_node',
                           'removed_genes': list()}

        # Collect all gene_ids of nodes that are not the input or output node (as they are unremovable) and
        # shuffle the list of those node ids for later random popping.
        node_count = 0
        bp_graph_node_ids = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                node_count += 1
                if gene.node != 1 and gene.node != 2:
                    bp_graph_node_ids.append(gene.gene_id)
        random.shuffle(bp_graph_node_ids)

        # Determine how many nodes will be removed based on the total node count
        number_of_nodes_to_rem = math.ceil(max_degree_of_mutation * node_count)

        # Remove nodes in loop until enough nodes are removed or until no node is left to be removed. When removing the
        # node, replace its incoming and outcoming connections with connections from each incoming node to each outgoing
        # node.
        rem_nodes_counter = 0
        while rem_nodes_counter < number_of_nodes_to_rem and len(bp_graph_node_ids) > 0:
            node_id_to_remove = bp_graph_node_ids.pop()
            node_to_remove = blueprint_graph[node_id_to_remove].node

            # Collect all gene ids with connections starting or ending in the chosen node, independent of if the node
            # is enabled or not (as this operation basically reverses the disabling of connections happening when adding
            # instead of removing a node), to be removed later. Also collect all end nodes of the outgoing connections
            # as well as all start nodes of all incoming connections.
            conn_ids_to_remove = list()
            conn_replacement_start_nodes = list()
            conn_replacement_end_nodes = list()
            for gene in blueprint_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintConn):
                    if gene.conn_start == node_to_remove:
                        conn_ids_to_remove.append(gene.gene_id)
                        conn_replacement_end_nodes.append(gene.conn_end)
                    elif gene.conn_end == node_to_remove:
                        conn_ids_to_remove.append(gene.gene_id)
                        conn_replacement_start_nodes.append(gene.conn_start)

            # Remove chosen node and all connections starting or ending in that node from blueprint graph
            del blueprint_graph[node_id_to_remove]
            parent_mutation['removed_genes'].append(node_id_to_remove)
            for id_to_remove in conn_ids_to_remove:
                del blueprint_graph[id_to_remove]
                parent_mutation['removed_genes'].append(id_to_remove)

            # Collect all current connections in blueprint graph to be checked against when creating new connections,
            # in case the connection already exists. This has be done in each iteration as those connections change
            # significantly for each round.
            bp_graph_conns = dict()
            for gene in blueprint_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintConn):
                    bp_graph_conns[(gene.conn_start, gene.conn_end, gene.enabled)] = gene.gene_id

            # Recreate the connections of the removed node by connecting all start nodes of the incoming connections to
            # all end nodes of the outgoing connections. Only recreate the connection if the connection is not already
            # present or if the connection present is disabled
            for new_start_node in conn_replacement_start_nodes:
                for new_end_node in conn_replacement_end_nodes:
                    # Check if a disabled variant of the connection to recreate is in the bp_graph. If so reenable it.
                    if (new_start_node, new_end_node, False) in bp_graph_conns:
                        conn_id_to_reenable = bp_graph_conns[(new_start_node, new_end_node, False)]
                        blueprint_graph[conn_id_to_reenable].set_enabled(True)
                    # Check if a no variant of the connection to recreate is in the bp_graph. If so, create it.
                    if (new_start_node, new_end_node, True) not in bp_graph_conns:
                        gene_id, gene = self.encoding.create_blueprint_conn(conn_start=new_start_node,
                                                                            conn_end=new_end_node)
                        blueprint_graph[gene_id] = gene

        # Create and return the offspring blueprint with the edited blueprint graph having removed nodes which were
        # replaced by a full connection between all incoming and all outgoing nodes.
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_mutated_blueprint_node_spec(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'node_spec',
                           'mutated_node_spec': dict()}

        # Identify all non-Input nodes in the blueprint graph by gene ID as the species of those can be mutated
        bp_graph_node_ids = set()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode) and gene.node != 1:
                bp_graph_node_ids.add(gene.gene_id)

        # Determine the node ids that have their species changed and the available module species to change into
        number_of_node_species_to_change = math.ceil(max_degree_of_mutation * len(bp_graph_node_ids))
        node_ids_to_change_species = random.sample(bp_graph_node_ids, k=number_of_node_species_to_change)
        available_mod_species = tuple(self.mod_species.keys())

        # Traverse through all randomly chosen node ids and change their module species randomly to one of the available
        for node_id_to_change_species in node_ids_to_change_species:
            former_node_species = blueprint_graph[node_id_to_change_species].species
            parent_mutation['mutated_node_spec'][node_id_to_change_species] = former_node_species
            blueprint_graph[node_id_to_change_species].species = random.choice(available_mod_species)

        # Create and return the offspring blueprint with the edited blueprint graph having mutated species
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_mutated_blueprint_optimizer(self, parent_blueprint):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()
        parent_opt_params = optimizer_factory.get_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'optimizer',
                           'mutated_params': parent_opt_params}

        # Randomly choose type of offspring optimizer and declare container collecting the specific parameters of
        # the offspring optimizer, setting only the chosen optimizer class
        offspring_optimizer_type = random.choice(self.available_optimizers)
        available_opt_params = self.available_opt_params[offspring_optimizer_type]
        offspring_opt_params = {'class_name': offspring_optimizer_type, 'config': dict()}

        if offspring_optimizer_type == parent_opt_params['class_name']:
            ## Mutation of the existing optimizers' parameters ##
            # Traverse each possible parameter option and determine a uniformly random value if its a categorical param
            # or try perturbing the the parent parameter if it is a sortable.
            for opt_param, opt_param_val_range in available_opt_params.items():
                # If the optimizer parameter is a categorical value choose randomly from the list
                if isinstance(opt_param_val_range, list):
                    offspring_opt_params['config'][opt_param] = random.choice(opt_param_val_range)
                # If the optimizer parameter is sortable, create a random value between the min and max values adhering
                # to the configured step
                elif isinstance(opt_param_val_range, dict):
                    if isinstance(opt_param_val_range['min'], int) \
                            and isinstance(opt_param_val_range['max'], int) \
                            and isinstance(opt_param_val_range['step'], int):
                        perturbed_param = int(np.random.normal(loc=parent_opt_params['config'][opt_param],
                                                               scale=opt_param_val_range['stddev']))
                        chosen_opt_param = round_with_step(perturbed_param,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    elif isinstance(opt_param_val_range['min'], float) \
                            and isinstance(opt_param_val_range['max'], float) \
                            and isinstance(opt_param_val_range['step'], float):
                        perturbed_param = np.random.normal(loc=parent_opt_params['config'][opt_param],
                                                           scale=opt_param_val_range['stddev'])
                        chosen_opt_param = round(round_with_step(perturbed_param,
                                                                 opt_param_val_range['min'],
                                                                 opt_param_val_range['max'],
                                                                 opt_param_val_range['step']), 4)
                    else:
                        raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                                  f"optimizer section is of type dict though the dict values are not "
                                                  f"of type int or float")
                    offspring_opt_params['config'][opt_param] = chosen_opt_param
                # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
                # parameter being set to True
                elif isinstance(opt_param_val_range, float):
                    offspring_opt_params['config'][opt_param] = random.random() < opt_param_val_range
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                              f"optimizer section is not one of the valid types of list, dict or float")

        else:
            ## Creation of a new optimizer with random parameters ##
            # Traverse each possible parameter option and determine a uniformly random value depending on if its a
            # categorical, sortable or boolean value
            for opt_param, opt_param_val_range in available_opt_params.items():
                # If the optimizer parameter is a categorical value choose randomly from the list
                if isinstance(opt_param_val_range, list):
                    offspring_opt_params['config'][opt_param] = random.choice(opt_param_val_range)
                # If the optimizer parameter is sortable, create a random value between the min and max values adhering
                # to the configured step
                elif isinstance(opt_param_val_range, dict):
                    if isinstance(opt_param_val_range['min'], int) \
                            and isinstance(opt_param_val_range['max'], int) \
                            and isinstance(opt_param_val_range['step'], int):
                        opt_param_random = random.randint(opt_param_val_range['min'],
                                                          opt_param_val_range['max'])
                        chosen_opt_param = round_with_step(opt_param_random,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    elif isinstance(opt_param_val_range['min'], float) \
                            and isinstance(opt_param_val_range['max'], float) \
                            and isinstance(opt_param_val_range['step'], float):
                        opt_param_random = random.uniform(opt_param_val_range['min'],
                                                          opt_param_val_range['max'])
                        chosen_opt_param = round(round_with_step(opt_param_random,
                                                                 opt_param_val_range['min'],
                                                                 opt_param_val_range['max'],
                                                                 opt_param_val_range['step']), 4)
                    else:
                        raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                                  f"optimizer section is of type dict though the dict values are not "
                                                  f"of type int or float")
                    offspring_opt_params['config'][opt_param] = chosen_opt_param
                # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
                # parameter being set to True
                elif isinstance(opt_param_val_range, float):
                    offspring_opt_params['config'][opt_param] = random.random() < opt_param_val_range
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                              f"optimizer section is not one of the valid types of list, dict or float")

        # Create new optimizer through encoding, having either the parent perturbed offspring parameters or randomly
        # new created parameters
        optimizer_factory = self.encoding.create_optimizer_factory(optimizer_parameters=offspring_opt_params)

        # Create and return the offspring blueprint with identical blueprint graph and modified optimizer_factory
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory,
                                              parent_mutation=parent_mutation)

    def _create_crossed_over_blueprint(self, parent_bp_1, parent_bp_2):
        """"""
        # Copy the parameters of both parent blueprints for the offspring
        bp_graph_1, opt_factory_1 = parent_bp_1.copy_parameters()
        bp_graph_2, opt_factory_2 = parent_bp_2.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': (parent_bp_1.get_id(), parent_bp_2.get_id()),
                           'mutation': 'crossover',
                           'gene_parent': dict(),
                           'optimizer_parent': None}

        # Create quickly searchable sets of gene ids to know about the overlap of genes in both blueprint graphs
        bp_graph_1_ids = set(bp_graph_1.keys())
        bp_graph_2_ids = set(bp_graph_2.keys())
        all_bp_graph_ids = bp_graph_1_ids.union(bp_graph_2_ids)

        # Create offspring blueprint graph by traversing all blueprint graph ids an copying over all genes that are
        # exclusive to either blueprint graph and randomly choosing the gene to copy over that is present in both graphs
        offspring_bp_graph = dict()
        for gene_id in all_bp_graph_ids:
            if gene_id in bp_graph_1_ids and gene_id in bp_graph_2_ids:
                if random.randint(0, 1) == 0:
                    offspring_bp_graph[gene_id] = bp_graph_1[gene_id]
                    parent_mutation['gene_parent'][gene_id] = parent_bp_1.get_id()
                else:
                    offspring_bp_graph[gene_id] = bp_graph_2[gene_id]
                    parent_mutation['gene_parent'][gene_id] = parent_bp_2.get_id()
            elif gene_id in bp_graph_1_ids:
                offspring_bp_graph[gene_id] = bp_graph_1[gene_id]
                parent_mutation['gene_parent'][gene_id] = parent_bp_1.get_id()
            else:  # if gene_id in bp_graph_2_ids
                offspring_bp_graph[gene_id] = bp_graph_2[gene_id]
                parent_mutation['gene_parent'][gene_id] = parent_bp_2.get_id()

        # For the optimizer factory choose the one from the fitter parent blueprint
        if parent_bp_1.get_fitness() > parent_bp_2.get_fitness():
            offspring_opt_factory = opt_factory_1
            parent_mutation['optimizer_parent'] = parent_bp_1.get_id()
        else:
            offspring_opt_factory = opt_factory_2
            parent_mutation['optimizer_parent'] = parent_bp_2.get_id()

        # Create and return the offspring blueprint with crossed over blueprint graph and optimizer_factory of the
        # fitter parent
        return self.encoding.create_blueprint(blueprint_graph=offspring_bp_graph,
                                              optimizer_factory=offspring_opt_factory,
                                              parent_mutation=parent_mutation)

    def save_population(self, save_dir_path):
        """"""
        # Set save file name as 'pop backup' and including the current generation
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        save_file_path = save_dir_path + f"population_backup_gen_{self.generation_counter}.json"

        # Serialize all modules for json output
        serialized_modules = dict()
        for mod_id, module in self.modules.items():
            serialized_modules[mod_id] = module.serialize()

        # Serialize all blueprints for json output
        serialized_blueprints = dict()
        for bp_id, blueprint in self.blueprints.items():
            serialized_blueprints[bp_id] = blueprint.serialize()

        # Use serialized module and blueprint population and extend it by algorithm internal evolution information
        serialized_population = {
            'best_genome_id': self.best_genome.get_id(),
            'modules': serialized_modules,
            'mod_species': self.mod_species,
            'mod_species_type': self.mod_species_type,
            'mod_species_counter': self.mod_species_counter,
            'blueprints': serialized_blueprints,
            'bp_species': self.bp_species,
            'bp_species_counter': self.bp_species_counter
        }

        # Backup the genotype of the best genome
        self.best_genome.save_genotype(save_dir_path=save_dir_path)

        # Actually save the just serialzied population as a json file
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_population, save_file, indent=4)
        print(f"Saved CoDeepNEAT population to file: {save_file_path}")

    def get_best_genome(self) -> CoDeepNEATGenome:
        """"""
        return self.best_genome
