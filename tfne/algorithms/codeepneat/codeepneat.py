import os
import sys
import json
import random
import statistics

from absl import logging
import tensorflow as tf

import tfne
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.codeepneat import CoDeepNEATGenome

from ._codeepneat_config_processing import CoDeepNEATConfigProcessing
from ._codeepneat_initialization import CoDeepNEATInitialization
from ._codeepneat_selection_mod import CoDeepNEATSelectionMOD
from ._codeepneat_selection_bp import CoDeepNEATSelectionBP
from ._codeepneat_evolution_mod import CoDeepNEATEvolutionMOD
from ._codeepneat_evolution_bp import CoDeepNEATEvolutionBP
from ._codeepneat_speciation_mod import CoDeepNEATSpeciationMOD
from ._codeepneat_speciation_bp import CoDeepNEATSpeciationBP


class CoDeepNEAT(BaseNeuroevolutionAlgorithm,
                 CoDeepNEATConfigProcessing,
                 CoDeepNEATInitialization,
                 CoDeepNEATSelectionMOD,
                 CoDeepNEATSelectionBP,
                 CoDeepNEATEvolutionMOD,
                 CoDeepNEATEvolutionBP,
                 CoDeepNEATSpeciationMOD,
                 CoDeepNEATSpeciationBP):
    """
    TFNE's implementation of the CoDeepNEAT algorithm.
    See the paper: https://arxiv.org/abs/1703.00548
    """

    def __init__(self, config, initial_state_file_path=None):
        """
        Initialize the CoDeepNEAT algorithm by processing and sanity checking the supplied configuration, which saves
        all algorithm config parameters as instance variables. Then initialize the CoDeepNEAT encoding and population.
        Alternatively, if a backup-state is supplied, reinitialize the encoding and population with the state present
        in that backup.
        @param config: ConfigParser instance holding all documentation specified sections of the CoDeepNEAT algorithm
        @param initial_state_file_path: string file path to a state backup that is to be resumed
        """
        # Register and process the supplied configuration
        self.config = config
        self._process_config()
        self._sanity_check_config()

        # Declare variables of environment shapes to which the created genomes have to adhere to
        self.input_shape = None
        self.output_shape = None

        # If an initial state of the evolution was supplied, load and recreate this state for the algorithm as well as
        # its dependencies
        if initial_state_file_path is not None:
            # Load the backed up state for the algorithm from file
            with open(initial_state_file_path) as saved_state_file:
                saved_state = json.load(saved_state_file)

            # Initialize and register an associated CoDeepNEAT encoding and population outfitted with the saved state
            self.enc = tfne.deserialization.load_encoding(serialized_encoding=saved_state['encoding'],
                                                          dtype=self.dtype)
            self.pop = tfne.deserialization.load_population(serialized_population=saved_state['population'],
                                                            dtype=self.dtype,
                                                            module_config_params=self.available_mod_params)
        else:
            # Initialize and register a blank associated CoDeepNEAT encoding and population
            self.enc = tfne.encodings.CoDeepNEATEncoding(dtype=self.dtype)
            self.pop = tfne.populations.CoDeepNEATPopulation()

    def initialize_population(self, environment):
        """
        Initialize the population according to the CoDeepNEAT algorithm. Initialize the module population by letting
        them all self initialize with random variables and assigning them to one species, as first occurs in the
        evolution. Initialize the blueprint population with each blueprint having a minimal graph and having the species
        1 as the node species. Also assign all created blueprints to one species.
        @param environment: instance of the evaluation environment
        """
        # If population already initialized, summarize status and abort initialization
        if self.pop.generation_counter is not None:
            print("Using supplied pre-evolved population. Supplied population summary:")
            print("Generation:       {:>4} || Best Genome Fitness: {:>8}\n"
                  "Modules count:    {:>4} || Mod species count:   {:>4}\n"
                  "Blueprints count: {:>4} || BP species count:    {:>4}\n"
                  .format(self.pop.generation_counter, self.pop.best_fitness,
                          len(self.pop.modules), len(self.pop.mod_species),
                          len(self.pop.blueprints), len(self.pop.bp_species)))
            return

        # No pre-evolved population supplied. Initialize it from scratch
        print("Initializing a new population of {} blueprints and {} modules..."
              .format(self.bp_pop_size, self.mod_pop_size))

        # Determine input and output shape parameters of the environment to which the created genomes of the population
        # have to adhere to
        self.input_shape = environment.get_input_shape()
        self.output_shape = environment.get_output_shape()

        # Set internal variables of the population to the initialization of a new population
        self.pop.generation_counter = 0
        self.pop.best_fitness = 0

        #### Initialize Module Population ####
        # Initialize module population with a basic speciation scheme, even when another speciation type is supplied
        # as config, only speciating modules according to their module type. Each module species (and therefore
        # module type) is initiated with the same amount of modules (or close to the same amount if module pop size
        # not evenly divisble). Parameters of all initial modules chosen as per module initialization implementation

        # Set initial species counter of basic speciation and initialize module species list
        self.pop.mod_species_counter = len(self.available_modules)
        for i in range(self.pop.mod_species_counter):
            spec_id = i + 1  # Start species counter with 1
            self.pop.mod_species[spec_id] = list()

        for i in range(self.mod_pop_size):
            # Decide on for which species a new module is added (uniformly distributed)
            chosen_species = (i % self.pop.mod_species_counter)

            # Determine type and the associated config parameters of chosen species and initialize a module with it
            mod_type = self.available_modules[chosen_species]
            mod_config_params = self.available_mod_params[mod_type]
            module_id, module = self.enc.create_initial_module(mod_type=mod_type,
                                                               config_params=mod_config_params)

            # Append newly created initial module to module container and to according species
            chosen_species_id = chosen_species + 1
            self.pop.modules[module_id] = module
            self.pop.mod_species[chosen_species_id].append(module_id)

            # Create a species representative if speciation method is not 'basic' and no representative chosen yet
            if self.mod_spec_type != 'basic' and chosen_species_id not in self.pop.mod_species_repr:
                self.pop.mod_species_repr[chosen_species_id] = module_id

        #### Initialize Blueprint Population ####
        # Initialize blueprint population with a minimal blueprint graph, only consisting of an input node (with
        # None species or the 'input' species respectively) and a single output node, having a randomly assigned
        # species. All hyperparameters of the blueprint are uniform randomly chosen. All blueprints are not
        # speciated in the beginning and are assigned to species 1.

        # Initialize blueprint species list and create tuple of the possible species the output node can take on
        self.pop.bp_species_counter = 1
        self.pop.bp_species[self.pop.bp_species_counter] = list()
        available_mod_species = tuple(self.pop.mod_species.keys())

        for _ in range(self.bp_pop_size):
            # Determine the module species of the initial (and only) node
            initial_node_species = random.choice(available_mod_species)

            # Initialize a new blueprint with minimal graph only using initial node species
            blueprint_id, blueprint = self._create_initial_blueprint(initial_node_species)

            # Append newly create blueprint to blueprint container and to only initial blueprint species
            self.pop.blueprints[blueprint_id] = blueprint
            self.pop.bp_species[self.pop.bp_species_counter].append(blueprint_id)

            # Create a species representative if speciation method is not 'basic' and no representative chosen yet
            if self.bp_spec_type != 'basic' and self.pop.bp_species_counter not in self.pop.bp_species_repr:
                self.pop.bp_species_repr[self.pop.bp_species_counter] = blueprint_id

    def evaluate_population(self, environment) -> (int, float):
        """
        Evaluate the population by building the specified amount of genomes from each blueprint, all having randomly
        assigned specific modules for the inherent blueprint module species. Set the evaluated fitness of each blueprint
        and each module as the average fitness achieved by all genomes in which the respective member was invovled in.
        Return the generational counter as well as the achieved fitness of the best genome.
        @param environment: instance of the evaluation environment
        @return: tuple of generation counter and best fitness achieved by best genome
        """
        # Create container collecting the fitness of the genomes that involve specific modules. Calculate the average
        # fitness of the genomes in which a module is involved in later and assign it as the module's fitness
        mod_fitnesses_in_genomes = dict()

        # Initialize population evaluation progress bar. Print notice of evaluation start
        genome_pop_size = len(self.pop.blueprints) * self.genomes_per_bp
        genome_eval_counter = 0
        genome_eval_counter_div = round(genome_pop_size / 40.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(genome_pop_size, self.pop.generation_counter))
        print_str = "\r[{:40}] {}/{} Genomes".format("", genome_eval_counter, genome_pop_size)
        sys.stdout.write(print_str)
        sys.stdout.flush()

        # Evaluate each blueprint independent from its species by building 'genomes_per_bp' genomes and averaging out
        # and assigning the resulting fitness
        for blueprint in self.pop.blueprints.values():
            # Get the species ids of all species present in the blueprint currently evaluated
            bp_module_species = blueprint.get_species()

            # Create container collecting the fitness of the genomes that involve that specific blueprint.
            bp_fitnesses_in_genomes = list()

            for _ in range(self.genomes_per_bp):
                # Assemble genome by first uniform randomly choosing a specific module from the species that the
                # blueprint nodes are referring to.
                bp_assigned_modules = dict()
                for i in bp_module_species:
                    chosen_module_id = random.choice(self.pop.mod_species[i])
                    bp_assigned_modules[i] = self.pop.modules[chosen_module_id]

                try:
                    # Create genome, using the specific blueprint, a dict of modules for each species, the configured
                    # output layers and input shape as well as the current generation
                    genome_id, genome = self.enc.create_genome(blueprint,
                                                               bp_assigned_modules,
                                                               self.output_layers,
                                                               self.input_shape,
                                                               self.pop.generation_counter)

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

                else:
                    # Now evaluate genome on registered environment and set its fitness
                    # NOTE: As CoDeepNEAT implementation currently only supports 1 eval instance, automatically choose
                    # that instance from the environment list
                    genome_fitness = environment.eval_genome_fitness(genome)
                    genome.set_fitness(genome_fitness)

                # Print population evaluation progress bar
                genome_eval_counter += 1
                progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
                print_str = "\r[{:40}] {}/{} Genomes | Genome ID {} achieved fitness of {}".format(
                    "=" * progress_mult,
                    genome_eval_counter,
                    genome_pop_size,
                    genome_id,
                    genome_fitness)
                sys.stdout.write(print_str)
                sys.stdout.flush()

                # Add newline after status update when debugging
                if logging.level_debug():
                    print("")

                # Assign the genome fitness to the blueprint and all modules used for the creation of the genome
                bp_fitnesses_in_genomes.append(genome_fitness)
                for assigned_module in bp_assigned_modules.values():
                    module_id = assigned_module.get_id()
                    if module_id in mod_fitnesses_in_genomes:
                        mod_fitnesses_in_genomes[module_id].append(genome_fitness)
                    else:
                        mod_fitnesses_in_genomes[module_id] = [genome_fitness]

                # Register genome as new best if it exhibits better fitness than the previous best
                if self.pop.best_fitness is None or genome_fitness > self.pop.best_fitness:
                    self.pop.best_genome = genome
                    self.pop.best_fitness = genome_fitness

                # Reset models, counters, layers, etc including in the GPU to avoid memory clutter from old models as
                # most likely only limited gpu memory is available
                tf.keras.backend.clear_session()

            # Average out collected fitness of genomes the blueprint was invovled in. Then assign that average fitness
            # to the blueprint
            bp_fitnesses_in_genomes_avg = round(statistics.mean(bp_fitnesses_in_genomes), 4)
            blueprint.set_fitness(bp_fitnesses_in_genomes_avg)

        # Average out collected fitness of genomes each module was invovled in. Then assign that average fitness to the
        # module
        for mod_id, mod_fitness_list in mod_fitnesses_in_genomes.items():
            mod_genome_fitness_avg = round(statistics.mean(mod_fitness_list), 4)
            self.pop.modules[mod_id].set_fitness(mod_genome_fitness_avg)

        # Calculate average fitness of each module species and add to pop.mod_species_fitness_history
        for spec_id, spec_mod_ids in self.pop.mod_species.items():
            spec_fitness_list = [self.pop.modules[mod_id].get_fitness() for mod_id in spec_mod_ids]
            spec_avg_fitness = round(statistics.mean(spec_fitness_list), 4)
            if spec_id in self.pop.mod_species_fitness_history:
                self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter] = spec_avg_fitness
            else:
                self.pop.mod_species_fitness_history[spec_id] = {self.pop.generation_counter: spec_avg_fitness}

        # Calculate average fitness of each blueprint species and add to pop.bp_species_fitness_history
        for spec_id, spec_bp_ids in self.pop.bp_species.items():
            spec_fitness_list = [self.pop.blueprints[bp_id].get_fitness() for bp_id in spec_bp_ids]
            spec_avg_fitness = round(statistics.mean(spec_fitness_list), 4)
            if spec_id in self.pop.bp_species_fitness_history:
                self.pop.bp_species_fitness_history[spec_id][self.pop.generation_counter] = spec_avg_fitness
            else:
                self.pop.bp_species_fitness_history[spec_id] = {self.pop.generation_counter: spec_avg_fitness}

        return self.pop.generation_counter, self.pop.best_fitness

    def summarize_population(self):
        """"""
        self.pop.summarize_population()

    def evolve_population(self) -> bool:
        """
        Evolve the population according to the CoDeepNEAT algorithm by first selecting all modules and blueprints, which
        eliminates low performing members and species and determines members elligible for being parents of offspring.
        Then evolve the module population by creating mutations or crossovers of elligible parents. Evolve the blueprint
        population by adding nodes or connections, removing nodes or connections, mutating module species or optimizers,
        etc. Subsequently speciate the module and blueprint population according to the chosen speciation method, which
        clusters the modules and blueprints according to their similarity.
        @return: bool flag, indicating ig population went extinct during evolution
        """
        #### Select Modules ####
        if self.mod_spec_type == 'basic':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_basic()
        elif self.mod_spec_type == 'param-distance-fixed':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_param_distance_fixed()
        elif self.mod_spec_type == 'param-distance-dynamic':
            mod_spec_offspring, mod_spec_parents, mod_spec_extinct = self._select_modules_param_distance_dynamic()
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        # If population went extinct abort evolution and return True
        if len(self.pop.mod_species) == 0:
            return True

        #### Select Blueprints ####
        if self.bp_spec_type == 'basic':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_basic()
        elif self.bp_spec_type == 'gene-overlap-fixed':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_gene_overlap_fixed()
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            bp_spec_offspring, bp_spec_parents = self._select_blueprints_gene_overlap_dynamic()
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        # If population went extinct abort evolution and return True
        if len(self.pop.bp_species) == 0:
            return True

        #### Evolve Modules ####
        new_module_ids = self._evolve_modules(mod_spec_offspring, mod_spec_parents)

        #### Evolve Blueprints ####
        new_bp_ids, bp_spec_parents = self._evolve_blueprints(bp_spec_offspring, bp_spec_parents, mod_spec_extinct)

        #### Speciate Modules ####
        if self.mod_spec_type == 'basic':
            self._speciate_modules_basic(mod_spec_parents, new_module_ids)
        elif self.mod_spec_type == 'param-distance-fixed':
            self._speciate_modules_param_distance_fixed(mod_spec_parents, new_module_ids)
        elif self.mod_spec_type == 'param-distance-dynamic':
            self._speciate_modules_param_distance_dynamic(mod_spec_parents, new_module_ids)
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        #### Speciate Blueprints ####
        if self.bp_spec_type == 'basic':
            self._speciate_blueprints_basic(bp_spec_parents, new_bp_ids)
        elif self.bp_spec_type == 'gene-overlap-fixed':
            self._speciate_blueprints_gene_overlap_fixed(bp_spec_parents, new_bp_ids)
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            self._speciate_blueprints_gene_overlap_dynamic(bp_spec_parents, new_bp_ids)
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        #### Return ####
        # Adjust generation counter and return False, signalling that the population has not gone extinct
        self.pop.generation_counter += 1
        return False

    def save_state(self, save_dir_path):
        """
        Save the state of the algorithm and the current evolutionary process by serializing all aspects to json
        compatible dicts and saving it as file to the supplied save dir path.
        @param save_dir_path: string of directory path to which the state should be saved
        """
        # Set save file name as 'pop backup' and including the current generation. Ensure that the save_dir_path exists
        # by creating the directories.
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        os.makedirs(save_dir_path, exist_ok=True)
        save_file_path = save_dir_path + f"tfne_state_backup_gen_{self.pop.generation_counter}.json"

        # Create serialized state of the evolutionary process. Set type of that serialized state.
        serialized_state = dict()
        serialized_state['type'] = 'CoDeepNEAT'

        # Create serialized population
        serialized_state['population'] = self.pop.serialize()

        # Create serialized encoding state
        serialized_state['encoding'] = self.enc.serialize()

        # Save the just serialized state as a json file
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_state, save_file, indent=4)
        print("Backed up generation {} of the CoDeepNEAT evolutionary run to file: {}"
              .format(self.pop.generation_counter,
                      save_file_path))

    def get_best_genome(self) -> CoDeepNEATGenome:
        """"""
        return self.pop.best_genome

    def get_eval_instance_count(self) -> int:
        """"""
        return 1
