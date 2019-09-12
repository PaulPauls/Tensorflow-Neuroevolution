import numpy as np
from absl import logging
from random import randint, random, choice

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm
from neuroevolution.encodings.direct.direct_encoding_gene import DirectEncodingGeneIDBank


class NEAT(BaseNeuroevolutionAlgorithm):
    def __init__(self, encoding, config):
        self.encoding = encoding

        # Declare and read in config parameters for the NEAT NE algorithm
        self.replacement_percentage = None
        self.mutate_prob = None
        self.recombine_prob = None
        self.mutate_weights_prob = None
        self.mutate_connection_prob = None
        self.mutate_node_prob = None
        self.activation_default = None
        self.activation_out = None
        self._read_config_parameters(config)

        # Check if mutate/recombine and different mutate probabilties are correct set and add up to 1
        assert self.mutate_prob + self.recombine_prob == 1.0
        assert self.mutate_weights_prob + self.mutate_connection_prob + self.mutate_node_prob == 1.0

        # As NEAT evolves model weights manually, disable automatic weight training
        self.trainable = False

    def _read_config_parameters(self, config):
        section_name_algorithm = 'NEAT' if config.has_section('NEAT') else 'NE_ALGORITHM'
        section_name_evolvable_encoding = 'DIRECT_ENCODING_EVOLVABLE' \
            if config.has_section('DIRECT_ENCODING_EVOLVABLE') else 'ENCODING_EVOLVABLE'
        self.replacement_percentage = config.getfloat(section_name_algorithm, 'replacement_percentage')
        self.mutate_prob = config.getfloat(section_name_algorithm, 'mutate_prob')
        self.recombine_prob = config.getfloat(section_name_algorithm, 'recombine_prob')
        self.mutate_weights_prob = config.getfloat(section_name_algorithm, 'mutate_weights_prob')
        self.mutate_connection_prob = config.getfloat(section_name_algorithm, 'mutate_connection_prob')
        self.mutate_node_prob = config.getfloat(section_name_algorithm, 'mutate_node_prob')
        self.activation_default = config.get(section_name_evolvable_encoding, 'activation_default')
        self.activation_out = config.get(section_name_evolvable_encoding, 'activation_out')

        logging.debug("NEAT NE Algorithm read from config: replacement_percentage = {}"
                      .format(self.replacement_percentage))
        logging.debug("NEAT NE Algorithm read from config: mutate_prob = {}".format(self.mutate_prob))
        logging.debug("NEAT NE Algorithm read from config: recombine_prob = {}".format(self.recombine_prob))
        logging.debug("NEAT NE Algorithm read from config: mutate_weights_prob = {}".format(self.mutate_weights_prob))
        logging.debug("NEAT NE Algorithm read from config: mutate_connection_prob = {}"
                      .format(self.mutate_connection_prob))
        logging.debug("NEAT NE Algorithm read from config: mutate_node_prob = {}".format(self.mutate_node_prob))
        logging.debug("NEAT NE Algorithm read from config: activation_default = {}".format(self.activation_default))
        logging.debug("NEAT NE Algorithm read from config: activation_out = {}".format(self.activation_out))

    def create_initial_genome(self, input_shape, num_output):
        genotype = list()
        # Determine if multidimensional input vector (as this is not yet implemented
        if len(input_shape) == 1:
            num_input = input_shape[0]

            # Create a connection from each input node to each output node
            for in_node in range(1, num_input + 1):
                for out_node in range(num_input + 1, num_input + num_output + 1):
                    genotype.append((in_node, out_node))

            # Specify layer activation functions for genotype
            activations = {'out_activation': self.activation_out,
                           'default_activation': self.activation_default}

        else:
            raise NotImplementedError("Multidimensional Input vector not yet supported")

        new_initialized_genome = self.encoding.create_new_genome(genotype, activations, trainable=self.trainable)
        return new_initialized_genome

    def create_new_generation(self, population):
        intended_pop_size = population.get_pop_size()
        replacement_count = int(self.replacement_percentage * intended_pop_size)
        # Remove the in replacement_count specified amount of the worst performing members of the population
        for _ in range(replacement_count):
            worst_genome = population.get_worst_genome()
            population.remove_genome(worst_genome)

        # Create a mutated or recombined genome (probability of either one determined by cfg parameter
        # 'mutate_recombine_ratio'), which are based on random genomes still in the population back to the population.
        for _ in range(replacement_count):
            if random() < self.mutate_prob:
                # Create and append mutated genome
                genome_to_mutate = population.get_genome(randint(0, intended_pop_size - replacement_count - 1))
                mutated_genome = self._create_mutated_genome(genome_to_mutate)
                population.append_genome(mutated_genome)
            else:
                # Create and append recombined genome
                range_of_possible_genome_indexes = list(range(0, intended_pop_size - replacement_count))
                index_first_genome = choice(range_of_possible_genome_indexes)
                range_of_possible_genome_indexes.remove(index_first_genome)
                index_second_genome = choice(range_of_possible_genome_indexes)

                genome_to_recombine_1 = population.get_genome(index_first_genome)
                genome_to_recombine_2 = population.get_genome(index_second_genome)

                recombined_genome = self._create_recombined_genome(genome_to_recombine_1, genome_to_recombine_2)
                population.append_genome(recombined_genome)

        # Return count of successfully replaced/mutated genomes
        return replacement_count

    def _create_mutated_genome(self, genome):
        # Create the choice of weight, connection or node mutation
        mutate_choice = random()

        # Get genotype and activations, as all mutations require this info from the genome they are based on
        genotype, activations = genome.serialize()

        if mutate_choice < self.mutate_weights_prob:
            # Create a new genome by mutating the weights of the supplied genome and leaving the genotype and
            # activations untouched
            weights = genome.get_weights()
            stddev = 1.0
            for layer_weights_index in range(len(weights)):
                layer_weights = weights[layer_weights_index]
                shape = layer_weights.shape
                mean = np.mean(layer_weights)
                stddev_pre = np.std(layer_weights)
                stddev = stddev_pre if stddev_pre != 0.0 else stddev
                weights[layer_weights_index] = np.random.normal(loc=mean, scale=stddev, size=shape)

            weight_mutated_genome = self.encoding.create_new_genome(genotype, activations, trainable=self.trainable)
            weight_mutated_genome.set_weights(weights)
            return weight_mutated_genome

        # Get topology_levels and access to unique gene IDs, as both connection and node mutations require this info
        topology_levels = genome.get_topology_levels()
        gene_id_bank = DirectEncodingGeneIDBank()

        if mutate_choice < self.mutate_weights_prob + self.mutate_connection_prob:
            # Create a new genome by adding a connection to the supplied genome
            for layer_index in range(len(topology_levels) - 1):
                possible_feedforward_nodes = set.union(*(topology_levels[layer_index + 1:]))
                for node in topology_levels[layer_index]:
                    for feedforward_node in possible_feedforward_nodes:
                        if (node, feedforward_node) not in genotype.values():
                            key = gene_id_bank.get_id((node, feedforward_node))
                            genotype[key] = (node, feedforward_node)
                            conn_mutated_genome = self.encoding.create_new_genome(genotype, activations,
                                                                                  trainable=self.trainable)
                            # ToDo: conn_mutated_genome.set_weights(...)
                            return conn_mutated_genome

        # If mutation choice fell to node mutation or it wasn't possible to add a connection, create a new genome by
        # adding a connection node to the supplied genome
        output_node_layer = randint(1, len(topology_levels) - 1)
        input_node_layer = randint(0, output_node_layer - 1)

        output_node = choice(tuple(topology_levels[output_node_layer]))
        input_node = choice(tuple(topology_levels[input_node_layer]))
        new_node = max(set.union(*topology_levels)) + 1

        key_in_new = gene_id_bank.get_id((input_node, new_node))
        key_new_out = gene_id_bank.get_id((new_node, output_node))
        genotype[key_in_new] = (input_node, new_node)
        genotype[key_new_out] = (new_node, output_node)

        node_mutated_genome = self.encoding.create_new_genome(genotype, activations, trainable=self.trainable)
        # ToDo: node_mutated_genome.set_weights(...)
        return node_mutated_genome

    def _create_recombined_genome(self, genome_1, genome_2):
        genotype_1, activations_1 = genome_1.serialize()
        genotype_2, activations_2 = genome_2.serialize()

        genotype = {**genotype_1, **genotype_2}
        assert activations_1 == activations_2

        recombined_genome = self.encoding.create_new_genome(genotype, activations_1, trainable=self.trainable)
        # ToDo: recombined_genome.set_weights(...)
        return recombined_genome
