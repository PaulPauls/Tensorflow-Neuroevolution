import math
import numpy as np
import tensorflow as tf
from absl import logging
from copy import deepcopy
from random import choice, random, randint, shuffle

from ..base_algorithm import BaseNeuroevolutionAlgorithm


class NEAT(BaseNeuroevolutionAlgorithm):
    def __init__(self, encoding, config):
        self.encoding = encoding

        # Declare and read in config parameters for the NEAT NE algorithm
        self.reproducing_fraction = None
        self.recombine_prob = None
        self.mutate_weights_prob = None
        self.add_conn_prob = None
        self.add_node_prob = None
        self.initial_connection = None
        self.species_elitism = None
        self.species_max_stagnation = None
        self.species_clustering = None
        self.activation_default = None
        self.activation_out = None
        self._read_config_parameters(config)
        self._log_class_parameters()

        assert self.recombine_prob + self.mutate_weights_prob + self.add_conn_prob + self.add_node_prob == 1.0

        # As NEAT evolves model weights manually, disable automatic weight training
        self.trainable = False
        self.species_id_counter = 0
        self.species_assignment = dict()
        self.species_representatives = dict()
        self.species_avg_fitness_history = dict()

    def _read_config_parameters(self, config):
        section_name_algorithm = 'NEAT' if config.has_section('NEAT') else 'NE_ALGORITHM'
        section_name_evolvable_encoding = 'DIRECT_ENCODING_EVOLVABLE' \
            if config.has_section('DIRECT_ENCODING_EVOLVABLE') else 'ENCODING_EVOLVABLE'

        self.reproducing_fraction = config.getfloat(section_name_algorithm, 'reproducing_fraction')
        self.recombine_prob = config.getfloat(section_name_algorithm, 'recombine_prob')
        self.mutate_weights_prob = config.getfloat(section_name_algorithm, 'mutate_weights_prob')
        self.add_conn_prob = config.getfloat(section_name_algorithm, 'add_conn_prob')
        self.add_node_prob = config.getfloat(section_name_algorithm, 'add_node_prob')
        self.initial_connection = config.get(section_name_algorithm, 'initial_connection')
        self.species_elitism = config.getint(section_name_algorithm, 'species_elitism')
        self.species_max_stagnation = config.get(section_name_algorithm, 'species_max_stagnation')
        self.species_clustering = config.get(section_name_algorithm, 'species_clustering')
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

    def _log_class_parameters(self):
        logging.debug("NEAT NE Algorithm parameter: encoding = {}".format(self.encoding.__class__.__name__))
        logging.debug("NEAT NE Algorithm read from config: reproducing_fraction = {}".format(self.reproducing_fraction))
        logging.debug("NEAT NE Algorithm read from config: recombine_prob = {}".format(self.recombine_prob))
        logging.debug("NEAT NE Algorithm read from config: mutate_weights_prob = {}".format(self.mutate_weights_prob))
        logging.debug("NEAT NE Algorithm read from config: add_conn_prob = {}".format(self.add_conn_prob))
        logging.debug("NEAT NE Algorithm read from config: add_node_prob = {}".format(self.add_node_prob))
        logging.debug("NEAT NE Algorithm read from config: initial_connection = {}".format(self.initial_connection))
        logging.debug("NEAT NE Algorithm read from config: species_elitism = {}".format(self.species_elitism))
        logging.debug("NEAT NE Algorithm read from config: species_max_stagnation = {}"
                      .format(self.species_max_stagnation))
        logging.debug("NEAT NE Algorithm read from config: species_clustering = {}".format(self.species_clustering))
        logging.debug("NEAT NE Algorithm read from config: activation_default = {}".format(self.activation_default))
        logging.debug("NEAT NE Algorithm read from config: activation_out = {}".format(self.activation_out))

    def initialize_population(self, population, initial_pop_size, input_shape, num_output):
        if len(input_shape) == 1:
            num_input = input_shape[0]
            genotype = list()

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

                    new_genome = self.encoding.create_genome(genotype, self.trainable)
                    population.add_genome(new_genome)

            else:
                raise NotImplementedError("Non-'full' initial connection not yet supported")
        else:
            raise NotImplementedError("Multidimensional input vectors not yet supported")

    def evolve_population(self, population, pop_size_fixed):

        # Assert that pop size is not fixed as NEAT does not operate on fixed pop sizes and handling this parameter has
        # therefore not been implemented in NEAT
        assert not pop_size_fixed

        max_stag_dur = self.species_max_stagnation[0]
        non_stag_improv = self.species_max_stagnation[1]

        # ToDo: At the moment does the species_elitism param save the X most recently emerged species, not the X best
        #       performing species, when multiple species are removed. Possibly change that after discussion with Rezsa.
        # Determine stagnating species to remove
        species_count = len(self.species_assignment)
        species_to_remove = []
        for species_id, avg_fitness_history in self.species_avg_fitness_history.items():
            if species_count - len(species_to_remove) <= self.species_elitism:
                break

            if len(avg_fitness_history) >= max_stag_dur:
                avg_fitnes_over_stag_dur = sum(avg_fitness_history[-max_stag_dur:]) / max_stag_dur
                non_stag_fitness = avg_fitness_history[-max_stag_dur] * non_stag_improv
                if avg_fitnes_over_stag_dur < non_stag_fitness:
                    logging.debug("Removng species {} as stagnating for {} generations..."
                                  .format(species_id, max_stag_dur))
                    species_to_remove.append(species_id)

        # Actually remove the determined stagnating species
        for species_id in species_to_remove:
            for genome_index in self.species_assignment[species_id]:
                population.delete_genome(genome_index)

            del self.species_assignment[species_id]
            del self.species_representatives[species_id]
            del self.species_avg_fitness_history[species_id]

        # Determine allotted offspring for each species
        species_allotted_offspring = dict()
        pop_size = population.get_pop_size()
        total_species_avg_fitness = 0
        for avg_fitness_history in self.species_avg_fitness_history.values():
            total_species_avg_fitness += avg_fitness_history[-1]

        for species_id in self.species_assignment:
            # Subtract 1 from the NEAT allotted offspring formula as NEAT has genome elitism of 1
            species_allotted_offspring[species_id] = math.ceil((self.species_avg_fitness_history[species_id][-1]
                                                                * pop_size) / total_species_avg_fitness) - 1

        # Determine indices of genomes of each species that are part of the X top percent of that species as specified
        # via the 'reproducing_fraction' parameter and are therefore elligible to be parents of the next generation
        species_repr_indices = dict()
        for species_id, species_genome_indices in self.species_assignment.items():
            sorted_genome_indices = sorted(species_genome_indices, key=lambda x: population.get_genome(x).get_fitness())
            repr_cutoff_index = int(len(sorted_genome_indices) * self.reproducing_fraction)
            if repr_cutoff_index == 0:
                # In case of species that consist of only 1 genome
                repr_cutoff_index = 1
            repr_genome_indices = sorted_genome_indices[-repr_cutoff_index:]
            species_repr_indices[species_id] = repr_genome_indices

        # Create new genomes through evolution to be added later
        new_genomes = list()
        mutate_weights_val = self.recombine_prob + self.mutate_weights_prob
        add_conn_val = mutate_weights_val + self.add_conn_prob
        for species_id, allotted_offspring in species_allotted_offspring.items():
            # Carry over the species chamption (which is saved as species representative) to new population as NEAT
            # has a genome elitism of 1
            new_genomes.append(population.get_genome(self.species_representatives[species_id]))
            # Adjust the mapping of the species representative to its location in the new genomes list
            self.species_representatives[species_id] = len(new_genomes) - 1
            for _ in range(allotted_offspring):
                parent_genome = population.get_genome(choice(species_repr_indices[species_id]))

                evolution_choice_val = random()
                if evolution_choice_val < self.recombine_prob:
                    # Out of simplicity currently possible that both parent genomes are the same
                    second_parent_genome = population.get_genome(choice(species_repr_indices[species_id]))
                    new_genomes.append(self._create_recombined_genome(parent_genome, second_parent_genome))
                elif evolution_choice_val < mutate_weights_val:
                    new_genomes.append(self._create_mutated_weights_genome(parent_genome))
                elif evolution_choice_val < add_conn_val:
                    new_genomes.append(self._create_added_conn_genome(parent_genome))
                else:
                    new_genomes.append(self._create_added_node_genome(parent_genome))

        # Replace the whole population with the newly created population
        population.replace_population(new_genomes)

    def _create_recombined_genome(self, parent_genome_1, parent_genome_2):
        genotype_1 = parent_genome_1.get_genotype()
        genotype_2 = parent_genome_2.get_genotype()
        new_genotype = deepcopy(genotype_1)

        genotype_1_gene_ids = []
        for gene in genotype_1:
            genotype_1_gene_ids.append(gene.gene_id)
        for gene in genotype_2:
            if gene.gene_id not in genotype_1_gene_ids:
                new_genotype.append(deepcopy(gene))

        return self.encoding.create_genome(new_genotype, self.trainable)

    def _create_mutated_weights_genome(self, parent_genome):
        new_genotype = deepcopy(parent_genome.get_genotype())
        for gene in new_genotype:
            try:
                gene.conn_weight = np.random.normal(loc=gene.conn_weight, scale=np.abs(gene.conn_weight) / 2)
            except AttributeError:
                gene.bias = np.random.normal(loc=gene.bias, scale=np.abs(gene.bias) / 2)

        return self.encoding.create_genome(new_genotype, self.trainable)

    def _create_added_conn_genome(self, parent_genome):
        topology_levels = parent_genome.get_topology_levels()
        new_genotype = deepcopy(parent_genome.get_genotype())

        existing_genotype_connections = []
        for gene in new_genotype:
            try:
                existing_genotype_connections.append((gene.conn_in, gene.conn_out))
            except AttributeError:
                pass

        highest_index_conn_in = len(topology_levels) - 1
        possible_conn_ins = list(set.union(*topology_levels[:highest_index_conn_in]))
        shuffle(possible_conn_ins)

        new_gene_conn = None
        for conn_in in possible_conn_ins:
            min_index_conn_out = None
            for layer_index in range(highest_index_conn_in):
                if conn_in in topology_levels[layer_index]:
                    min_index_conn_out = layer_index + 1
            possible_conn_outs = list(set.union(*topology_levels[min_index_conn_out:]))
            shuffle(possible_conn_outs)
            for conn_out in possible_conn_outs:
                if (conn_in, conn_out) not in existing_genotype_connections:
                    new_gene_conn = self.encoding.create_gene_connection(conn_in, conn_out)
                    new_genotype.append(new_gene_conn)
                    break
            if new_gene_conn is not None:
                break

        return self.encoding.create_genome(new_genotype, self.trainable)

    def _create_added_node_genome(self, parent_genome):
        topology_levels = parent_genome.get_topology_levels()
        new_genotype = deepcopy(parent_genome.get_genotype())

        new_node = max(set.union(*topology_levels)) + 1
        index_new_conn_in = randint(0, len(topology_levels) - 2)
        index_new_conn_out = randint(index_new_conn_in + 1, len(topology_levels) - 1)
        new_conn_in = choice(tuple(topology_levels[index_new_conn_in]))
        new_conn_out = choice(tuple(topology_levels[index_new_conn_out]))

        new_gene_node = self.encoding.create_gene_node(new_node, self.activation_default)
        new_gene_conn_in_node = self.encoding.create_gene_connection(new_conn_in, new_node)
        new_gene_conn_node_out = self.encoding.create_gene_connection(new_node, new_conn_out)
        new_genotype.append(new_gene_node)
        new_genotype.append(new_gene_conn_in_node)
        new_genotype.append(new_gene_conn_node_out)

        return self.encoding.create_genome(new_genotype, self.trainable)

    def evaluate_population(self, population, genome_eval_function):
        for i in range(population.get_pop_size()):
            genome = population.get_genome(i)
            if genome.get_fitness() == 0:
                genome.set_fitness(genome_eval_function(genome))

        # Speciate population by first clustering it and then applying fitness sharing
        self._cluster_population(population)
        self._apply_fitness_sharing(population)

    def _cluster_population(self, population):
        # If no species exist, set the first genome in the population as the representative of the first species
        if not self.species_representatives:
            self.species_id_counter += 1
            self.species_representatives[self.species_id_counter] = 0
            self.species_avg_fitness_history[self.species_id_counter] = []

        self.species_assignment = dict()
        for species_id in self.species_representatives:
            self.species_assignment[species_id] = []

        distance = dict()
        distance_threshold = self.species_clustering[1]
        assert self.species_clustering[0] == 'threshold-fixed'

        for i in range(population.get_pop_size()):
            genome = population.get_genome(i)
            for species_id, repr_genome_index in self.species_representatives.items():
                repr_genome = population.get_genome(repr_genome_index)
                distance[species_id] = self._calculate_genome_distance(genome, repr_genome)
            closest_species_id = min(distance, key=distance.get)
            smallest_distance = distance[closest_species_id]

            if smallest_distance <= distance_threshold:
                self.species_assignment[closest_species_id].append(i)
            else:
                self.species_id_counter += 1
                self.species_representatives[self.species_id_counter] = i
                self.species_assignment[self.species_id_counter] = [i]
                self.species_avg_fitness_history[self.species_id_counter] = []

        # Assign best genome of each species as the new species_representative
        for species_id, species_genome_indices in self.species_assignment.items():
            best_genome_index = max(species_genome_indices, key=lambda x: population.get_genome(x).get_fitness())
            self.species_representatives[species_id] = best_genome_index

    def _calculate_genome_distance(self, genome_1, genome_2):
        gene_ids_1 = genome_1.get_gene_ids()
        gene_ids_2 = genome_2.get_gene_ids()
        return len(gene_ids_1.symmetric_difference(gene_ids_2))

    def _apply_fitness_sharing(self, population):
        # Apply fitness sharing AND log species average fitness as all calculations are performed here anyway
        for species_id, species_genome_indices in self.species_assignment.items():
            species_size = len(species_genome_indices)
            fitness_sum = 0
            for genome_index in species_genome_indices:
                genome = population.get_genome(genome_index)
                adjusted_fitness = round(genome.get_fitness() / species_size, 3)
                genome.set_adj_fitness(adjusted_fitness)
                fitness_sum += adjusted_fitness
            species_avg_fitness = round(fitness_sum / species_size, 3)
            self.species_avg_fitness_history[species_id].append(species_avg_fitness)

    def summarize_population(self, population):
        generation = population.get_generation_counter()
        best_fitness = population.get_best_genome().get_fitness()
        average_fitness = population.get_average_fitness()
        pop_size = population.get_pop_size()
        species_count = len(self.species_assignment)
        logging.info("#### GENERATION: {:>4} ## BEST_FITNESS: {:>8} ## AVERAGE_FITNESS: {:>8} "
                     "## POP_SIZE: {:>4} ## SPECIES_COUNT: {:>4} ####"
                     .format(generation, best_fitness, average_fitness, pop_size, species_count))

        for species_id, species_genome_indices in self.species_assignment.items():
            species_best_fitness = population.get_genome(self.species_representatives[species_id]).get_fitness()
            species_avg_fitness = self.species_avg_fitness_history[species_id][-1]
            species_size = len(species_genome_indices)
            logging.info("---- SPECIES_ID: {:>4} -- SPECIES_BEST_FITNESS: {:>4} "
                         "-- SPECIES_AVG_FITNESS: {:>4} -- SPECIES_SIZE: {:>4} ----"
                         .format(species_id, species_best_fitness, species_avg_fitness, species_size))
            for genome_index in species_genome_indices:
                logging.debug(population.get_genome(genome_index))
