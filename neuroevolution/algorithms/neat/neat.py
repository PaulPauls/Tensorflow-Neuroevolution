import tensorflow as tf
from absl import logging
from collections import deque

from ..base_algorithm import BaseNeuroevolutionAlgorithm


class NEAT(BaseNeuroevolutionAlgorithm):
    def __init__(self, encoding, config):
        self.encoding = encoding

        # Declare and read in config parameters for the NEAT NE algorithm
        self.genome_elitism = None
        self.reproduction_cutoff = None
        self.recombine_prob = None
        self.mutate_weights_prob = None
        self.add_conn_prob = None
        self.add_node_prob = None
        self.initial_connection = None
        self.species_elitism = None
        self.species_min_size = None
        self.species_max_size = None
        self.species_max_stagnation = None
        self.species_clustering = None
        self.species_interbreeding = None
        self.activation_default = None
        self.activation_out = None
        self._read_config_parameters(config)

        assert self.recombine_prob + self.mutate_weights_prob + self.add_conn_prob + self.add_node_prob == 1.0

        # As NEAT evolves model weights manually, disable automatic weight training
        self.trainable = False

    def _read_config_parameters(self, config):
        section_name_algorithm = 'NEAT' if config.has_section('NEAT') else 'NE_ALGORITHM'
        section_name_evolvable_encoding = 'DIRECT_ENCODING_EVOLVABLE' \
            if config.has_section('DIRECT_ENCODING_EVOLVABLE') else 'ENCODING_EVOLVABLE'

        self.genome_elitism = config.getint(section_name_algorithm, 'genome_elitism')
        self.reproduction_cutoff = config.getfloat(section_name_algorithm, 'reproduction_cutoff')
        self.recombine_prob = config.getfloat(section_name_algorithm, 'recombine_prob')
        self.mutate_weights_prob = config.getfloat(section_name_algorithm, 'mutate_weights_prob')
        self.add_conn_prob = config.getfloat(section_name_algorithm, 'add_conn_prob')
        self.add_node_prob = config.getfloat(section_name_algorithm, 'add_node_prob')
        self.initial_connection = config.get(section_name_algorithm, 'initial_connection')
        self.species_elitism = config.getint(section_name_algorithm, 'species_elitism')
        self.species_min_size = config.getint(section_name_algorithm, 'species_min_size')
        self.species_max_size = config.getint(section_name_algorithm, 'species_max_size')
        self.species_max_stagnation = config.get(section_name_algorithm, 'species_max_stagnation')
        self.species_clustering = config.get(section_name_algorithm, 'species_clustering')
        self.species_interbreeding = config.getboolean(section_name_algorithm, 'species_interbreeding')
        self.activation_default = config.get(section_name_evolvable_encoding, 'activation_default')
        self.activation_out = config.get(section_name_evolvable_encoding, 'activation_out')

        if ',' in self.species_clustering:
            species_clustering_alg = self.species_clustering[:self.species_clustering.find(',')]
            species_clustering_val = float(self.species_clustering[self.species_clustering.find(',') + 2:])
            self.species_clustering = (species_clustering_alg, species_clustering_val)

        if ',' in self.species_max_stagnation:
            species_max_stagnation_duration = int(self.species_max_stagnation[:self.species_max_stagnation.find(',')])
            species_max_stagnation_perc = float(self.species_max_stagnation[self.species_max_stagnation.find(',') + 2:])
            self.species_max_stagnation = (species_max_stagnation_duration, species_max_stagnation_perc)

        self.activation_default = tf.keras.activations.deserialize(self.activation_default)
        self.activation_out = tf.keras.activations.deserialize(self.activation_out)

        logging.debug("NEAT NE Algorithm read from config: genome_elitism = {}".format(self.genome_elitism))
        logging.debug("NEAT NE Algorithm read from config: reproduction_cutoff = {}".format(self.reproduction_cutoff))
        logging.debug("NEAT NE Algorithm read from config: recombine_prob = {}".format(self.recombine_prob))
        logging.debug("NEAT NE Algorithm read from config: mutate_weights_prob = {}".format(self.mutate_weights_prob))
        logging.debug("NEAT NE Algorithm read from config: add_conn_prob = {}".format(self.add_conn_prob))
        logging.debug("NEAT NE Algorithm read from config: add_node_prob = {}".format(self.add_node_prob))
        logging.debug("NEAT NE Algorithm read from config: initial_connection = {}".format(self.initial_connection))
        logging.debug("NEAT NE Algorithm read from config: species_elitism = {}".format(self.species_elitism))
        logging.debug("NEAT NE Algorithm read from config: species_min_size = {}".format(self.species_min_size))
        logging.debug("NEAT NE Algorithm read from config: species_max_size = {}".format(self.species_max_size))
        logging.debug("NEAT NE Algorithm read from config: species_max_stagnation = {}"
                      .format(self.species_max_stagnation))
        logging.debug("NEAT NE Algorithm read from config: species_clustering = {}".format(self.species_clustering))
        logging.debug("NEAT NE Algorithm read from config: species_interbreeding = {}"
                      .format(self.species_interbreeding))
        logging.debug("NEAT NE Algorithm read from config: activation_default = {}".format(self.activation_default))
        logging.debug("NEAT NE Algorithm read from config: activation_out = {}".format(self.activation_out))

    def initialize_population(self, population, initial_pop_size, input_shape, num_output):
        if len(input_shape) == 1:
            num_input = input_shape[0]
            generation = population.get_generation_counter()
            genotype = deque()

            if self.initial_connection == 'full':
                for _ in range(initial_pop_size):
                    genotype.clear()

                    for conn_in in range(1, num_input + 1):
                        for conn_out in range(num_input + 1, num_input + num_output + 1):
                            new_gene_conn = self.encoding.create_gene_connection(conn_in, conn_out)
                            genotype.append(new_gene_conn)

                    for node in range(num_input + 1, num_input + num_output + 1):
                        new_gene_node = self.encoding.create_gene_node(node, self.activation_out)
                        genotype.append(new_gene_node)

                    new_genome = self.encoding.create_genome(genotype, self.trainable, 1, generation)
                    population.add_genome(1, new_genome)

            else:
                raise NotImplementedError("Non-'full' initial connection not yet supported")
        else:
            raise NotImplementedError("Multidimensional input vectors not yet supported")

    def evolve_population(self, population, pop_size_fixed):

        if pop_size_fixed:
            original_pop_size = population.get_pop_size()
            assert self.species_elitism * self.species_max_size >= original_pop_size

        max_stagnation_duration = self.species_max_stagnation[0]
        non_stagnation_improve_rate = 1 + self.species_max_stagnation[1]

        for species_id, species_avg_fitness_log in population.get_sorted_species_avg_fitness_log():
            if population.get_species_count() <= self.species_elitism:
                break

            if len(species_avg_fitness_log) >= max_stagnation_duration:
                average_avg_fitness = sum(species_avg_fitness_log[-max_stagnation_duration:]) / max_stagnation_duration
                non_stagnation_fitness = species_avg_fitness_log[-max_stagnation_duration] * non_stagnation_improve_rate
                if average_avg_fitness < non_stagnation_fitness:
                    logging.debug("Removing species {} as stagnating for {} generations..."
                                  .format(species_id, max_stagnation_duration))
                    population.remove_species(species_id)


        # ToDo

        if pop_size_fixed:
            genomes_to_add_min = int(original_pop_size / population.get_species_count()) - self.genome_elitism
        else:
            genomes_to_add_min = self.species_max_size - self.genome_elitism

        reproduction_indexes = dict()
        genomes_to_add = dict()

        for species_id in population.get_species_ids():
            species = population.get_species(species_id)

            reproduction_cutoff_abs = int(self.reproduction_cutoff * len(species))
            if reproduction_cutoff_abs < self.genome_elitism:
                reproduction_cutoff_abs = self.genome_elitism

            reproduction_genome_indices = get_genome_indices_in_species_sorted_by_fitness(species)[-reproduction_cutoff_abs:]

            reproduction_indexes[species_id] = reproduction_genome_indices

            genomes_to_add[species_id] = genomes_to_add_min
            if pop_size_fixed and population.get_species_count() * genomes_to_add_min + len(genomes_to_add) < original_pop_size:
                genomes_to_add[species_id] += 1


        new_genomes = dict()
        for species_id in population.get_species_ids():
            for _ in range(genomes_to_add[species_id]):
                genome_to_mutate = population.get_genome(species_id, choice(reproduction_indexes))

                random_val = random()

                mutate_weights_val = self.recombine_prob + self.mutate_weights_prob
                add_conn_val = mutate_weights_val + self.add_conn_prob

                if random_val < self.recombine_prob:
                    if self.species_interbreeding:
                        genome_to_recombine = choice(reproduction_indexes[choice(...)])
                    else:
                        genome_to_recombine = choice(reproduction_indexes[species_id])

                    new_genome = self._create_recombined_genome(genome_to_mutate, genome_to_recombine)
                elif random_val < mutate_weights_val:
                    new_genome = self._create_mutated_weights_genome(genome_to_mutate)
                elif random_val < add_conn_val:
                    new_genome = self._create_added_conn_genome(genome_to_mutate)
                else:
                    new_genome = self._create_added_node_genome(genome_to_mutate, self.activation_default)

                new_genomes[species_id] = new_genome

        for species_id in population.get_species_ids():
            population.preserve_x_genomes_and_replace_rest_with_y_genomes(species_id, self.genome_elitism, new_genomes[species_id])

        '''
        # ToDo: Recombine genomes
        self.genome_elitism: # of unaltered genomes for the next generation
        self.reproduction_cutoff: # the x fittest genomes are the basis for reproduction, mutation for the next gen
                                  Consider special cases of '0' and 'cutoff higher than genome elitism'
        self.recombine_prob: Prob of recombining chosen genome
        self.species_interbreeding: (if recombining)
        self.mutate_weights_prob: Prob of mutating weights of chosen genome
        self.add_conn_prob: Prob of add conn to chosen genome
        self.add_node_prob: Prob of adding node to chosen genome
        self.activation_default: (if adding node)
        self.species_max_size: If pop size not fixed, fill up species with mutations up to this point.
                               If pop size fixed: fill up #(genomes_removed/species_count) and add the extra genomes
                                    to the best performing species
        '''
        raise NotImplementedError()

    def speciate_population(self, population):

        '''
        relevant variables:
        self.genome_elitism
        self.reproduction_cutoff
        self.recombine_prob
        self.mutate_weights_prob
        self.add_conn_prob
        self.add_node_prob
        self.initial_connection
        self.species_elitism
        self.species_min_size
        self.species_max_size
        self.species_max_stagnation
        self.species_clustering
        self.species_interbreeding
        self.activation_default
        self.activation_out
        '''
        raise NotImplementedError()

    def uses_speciation(self):
        return True
