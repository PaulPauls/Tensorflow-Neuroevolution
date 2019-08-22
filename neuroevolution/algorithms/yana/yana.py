import tensorflow as tf
from random import randint, choice

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class YANA(BaseNeuroevolutionAlgorithm):
    """
    Implementation of a _dummy_ neuroevolution algorithm, named 'Yet Another Neuroevolution Algorithm'. This
    neuroevolution algorithm is designed to work with the direct encoding and direct-encoding genomes. The algorithm
    does all required neuroevolution tasks in the most basic way, only creating minimal fully connected genotype models
    when creating an initial genome and only adding nodes or connections to the best genome when evolving the
    population..
    """

    def __init__(self, encoding, config):
        self.logger = tf.get_logger()

        self.encoding = encoding

        # Read in config parameters for neuroevolution algorithm
        self.replacement_percentage = config.getfloat('NE_ALGORITHM', 'replacement_percentage')
        self.genome_default_activation = config.get('NE_ALGORITHM', 'default_activation')
        self.genome_out_activation = config.get('NE_ALGORITHM', 'out_activation')
        self.logger.debug("NE Algorithm read from config: replacement_percentage = {}"
                          .format(self.replacement_percentage))
        self.logger.debug("NE Algorithm read from config: genome_default_activation = {}"
                          .format(self.genome_default_activation))
        self.logger.debug("NE Algorithm read from config: genome_out_activation = {}"
                          .format(self.genome_out_activation))

        # As YANA uses SGD to optimize the weights of the topology (and as of now only evolves topology, not weights),
        # set  trainable variable to true
        self.trainable = True

    def create_initial_genome(self, input_shape, num_output):
        """
        Create a single genome of the chosen encoding with a fully connected genotype model, connecting all inputs
        to all outputs. Return this genome.
        """
        genotype = dict()
        # Determine if multidimensional input vector (as this is not yet implemented
        if len(input_shape) == 1:
            num_input = input_shape[0]

            # Create a connection from each input node to each output node
            key_counter = 1
            for in_node in range(1, num_input + 1):
                for out_node in range(num_input + 1, num_input + num_output + 1):
                    conn_in_out = (in_node, out_node)
                    genotype[key_counter] = conn_in_out
                    key_counter += 1

            # Specify layer activation functions for genotype
            activations = {'out_activation': self.genome_out_activation,
                           'default_activation': self.genome_default_activation}

        else:
            raise NotImplementedError("Multidimensional Input vector not yet supported")

        new_initialized_genome = self.encoding.create_new_genome(genotype, activations, trainable=self.trainable)
        return new_initialized_genome

    def create_new_generation(self, population):
        """
        Create a new generation in the population by removing X percent (specified in 'self.replacement_percentage')
        of the population and replacing them with mutated genomes, which are based on randomly chosen genomes that are
        left in the population.
        """
        replacement_count = int(self.replacement_percentage * population.get_pop_size())
        # Remove the in replacement_count specified amount of the worst performing members of the population
        for _ in range(replacement_count):
            worst_genome = population.get_worst_genome()
            population.remove_genome(worst_genome)

        # Add the same number of mutated genomes (mutated from random genomes still in pop) back to the population
        for _ in range(replacement_count):
            genome_to_mutate = population.get_genome(randint(0, population.get_pop_size() - replacement_count - 1))
            mutated_genome = self._create_mutated_genome(genome_to_mutate)
            population.append_genome(mutated_genome)

        # Return count of successfully replaced/mutated genomes
        return replacement_count

    def _create_mutated_genome(self, genome):
        """
        Create a mutated genome based on the supplied genome by copying its genotype and activations and having a
        50/50 chance of either adding a new node or a new connection to its genotype. If no new connection is possible
        because the genotype is fully connected, add a new node to the genotype. Then create a new genome from this
        adjusted genotype and the copied over activations and return it.
        """
        genotype, activations = genome.serialize()
        topology_levels = genome.get_topology_levels()

        # 50/50 chance of either adding a new connection or adding a new node to the existing genome
        add_connection = randint(0, 1)
        if add_connection:
            # Add new connection
            for layer_index in range(len(topology_levels) - 1):
                possible_feedforward_nodes = set.union(*(topology_levels[layer_index + 1:]))
                for node in topology_levels[layer_index]:
                    for feedforward_node in possible_feedforward_nodes:
                        if (node, feedforward_node) not in genotype.values():
                            key = max(genotype.keys()) + 1
                            genotype[key] = (node, feedforward_node)
                            mutated_genome = self.encoding.create_new_genome(genotype, activations,
                                                                             trainable=self.trainable)
                            return mutated_genome

        # if no possible connection to add was found, add a new node:
        output_node_layer = randint(1, len(topology_levels) - 1)
        input_node_layer = randint(0, output_node_layer - 1)

        output_node = choice(tuple(topology_levels[output_node_layer]))
        input_node = choice(tuple(topology_levels[input_node_layer]))
        new_node = max(set.union(*topology_levels)) + 1

        key = max(genotype.keys()) + 1
        genotype[key] = (input_node, new_node)
        genotype[key + 1] = (new_node, output_node)

        mutated_genome = self.encoding.create_new_genome(genotype, activations, trainable=self.trainable)
        return mutated_genome
