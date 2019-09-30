import numpy as np
import tensorflow as tf
from absl import logging
from copy import deepcopy
from collections import deque
from random import choice, random, shuffle

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

        if pop_size_fixed:
            genomes_to_add = int(original_pop_size / population.get_species_count()) - self.genome_elitism
        else:
            genomes_to_add = self.species_max_size - self.genome_elitism

        species_reproduction_indices = dict()
        species_genomes_to_add = dict()

        for species_id in population.get_species_ids():

            reproduction_cutoff_abs = int(self.reproduction_cutoff * population.get_species_length(species_id))
            if reproduction_cutoff_abs < self.genome_elitism:
                reproduction_cutoff_abs = self.genome_elitism

            species_reproduction_indices[species_id] = \
                population.get_fitness_sorted_indices_of_species_genomes(species_id)[-reproduction_cutoff_abs:]

            species_genomes_to_add[species_id] = genomes_to_add
            if pop_size_fixed:
                if population.get_species_count() * genomes_to_add + len(species_genomes_to_add) < original_pop_size:
                    species_genomes_to_add[species_id] += 1

        new_genomes = dict()
        generation = population.get_generation_counter()
        mutate_weights_val = self.recombine_prob + self.mutate_weights_prob
        add_conn_val = mutate_weights_val + self.add_conn_prob
        for species_id in population.get_species_ids():
            for _ in range(species_genomes_to_add[species_id]):
                genome_to_mutate = population.get_genome(species_id, choice(species_reproduction_indices[species_id]))
                random_val = random()

                if random_val < self.recombine_prob:
                    if self.species_interbreeding:
                        species_id_recombination = choice(population.get_species_ids())
                    else:
                        species_id_recombination = species_id

                    genome_index_recombination = choice(species_reproduction_indices[species_id_recombination])
                    genome_to_recombine = population.get_genome(species_id_recombination, genome_index_recombination)
                    new_genotype = self._create_recombined_genotype(genome_to_mutate, genome_to_recombine)
                elif random_val < mutate_weights_val:
                    new_genotype = self._create_mutated_weights_genotype(genome_to_mutate)
                elif random_val < add_conn_val:
                    new_genotype = self._create_added_conn_genotype(genome_to_mutate)
                else:
                    new_genotype = self._create_added_node_genotype(genome_to_mutate, self.activation_default)

                new_genome = self.encoding.create_genome(new_genotype, self.trainable, species_id, generation)
                new_genomes[species_id] = new_genome

        for species_id in population.get_species_ids():
            genomes_to_delete = \
                population.get_fitness_sorted_indices_of_species_genomes(species_id)[:-self.genome_elitism]
            for genome_index_to_delete in genomes_to_delete:
                population.remove_genome(species_id, genome_index_to_delete)

            for genome_to_add in new_genomes[species_id]:
                population.add_genome(species_id, genome_to_add)

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

    @staticmethod
    def uses_speciation():
        return True

    def _create_recombined_genotype(self, genome_1, genome_2):
        genotype_1 = genome_1.get_genotype()
        genotype_2 = genome_2.get_genotype()

        genotype_1_gene_ids = []
        for gene in genotype_1:
            genotype_1_gene_ids.append(gene.gene_id)

        new_genotype = deepcopy(genotype_1)
        for gene in genotype_2:
            if gene.gene_id not in genotype_1_gene_ids:
                new_genotype.append(deepcopy(gene))

        return new_genotype

    def _create_mutated_weights_genotype(self, genome):
        new_genotype = deepcopy(genome.get_genotype())
        for gene in new_genotype:
            try:
                gene.conn_weight = np.random.normal(loc=gene.conn_weight, scale=np.abs(gene.conn_weight) / 2)
            except AttributeError:
                gene.bias = np.random.normal(loc=gene.bias, scale=np.abs(gene.bias) / 2)

        return new_genotype

    def _create_added_conn_genotype(self, genome):
        topology_levels = genome.get_topology_levels()
        new_genotype = deepcopy(genome.get_genotype())

        existing_genotype_connections = []
        for gene in new_genotype:
            try:
                existing_genotype_connections.append((gene.conn_in, gene.conn_out))
            except AttributeError:
                pass

        highest_index_origin_node = len(topology_levels) - 1
        possible_origin_nodes = list(set.union(*topology_levels[:highest_index_origin_node]))
        shuffle(possible_origin_nodes)

        new_gene_conn = None
        for origin_node in possible_origin_nodes:
            min_index_forward_node = None
            for layer_index in range(highest_index_origin_node):
                if origin_node in topology_levels[layer_index]:
                    min_index_forward_node = layer_index + 1
            possible_forward_nodes = list(set.union(*topology_levels[min_index_forward_node:]))
            shuffle(possible_forward_nodes)
            for forward_node in possible_forward_nodes:
                if (origin_node, forward_node) not in existing_genotype_connections:
                    new_gene_conn = self.encoding.create_gene_connection(origin_node, forward_node)
                    new_genotype.append(new_gene_conn)
                    break
            if new_gene_conn is not None:
                break

        return new_genotype

    def _create_added_node_genotype(self, genome, activation):
        raise NotImplementedError()
