import tensorflow as tf
from random import randint, random, choice

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class NEAT(BaseNeuroevolutionAlgorithm):
    """
    ToDo: Class docstring
    """

    def __init__(self, encoding, config):
        self.logger = tf.get_logger()

        self.encoding = encoding

        # Read in config parameters for the NEAT neuroevolution algorithm and the evolvable config parameters of the
        # Direct encoding
        section_name_algorithm = 'NEAT' if config.has_section('NEAT') else 'NE_ALGORITHM'
        section_name_evolvable_encoding = 'DIRECT_ENCODING_EVOLVABLE' \
            if config.has_section('DIRECT_ENCODING_EVOLVABLE') else 'ENCODING_EVOLVABLE'
        self.replacement_percentage = config.getfloat(section_name_algorithm, 'replacement_percentage')
        self.mutate_prob = config.getfloat(section_name_algorithm, 'mutate_prob')
        self.recombine_prob = config.getfloat(section_name_algorithm, 'recombine_prob')
        self.mutate_weights_prob = config.getfloat(section_name_algorithm, 'mutate_weights_prob')
        self.mutate_connection_prob = config.getfloat(section_name_algorithm, 'mutate_connection_prob')
        self.mutate_node_prob = config.getfloat(section_name_algorithm, 'mutate_node_prob')
        self.genome_default_activation = config.get(section_name_evolvable_encoding, 'default_activation')
        self.genome_out_activation = config.get(section_name_evolvable_encoding, 'out_activation')

        self.logger.debug("NEAT NE Algorithm read from config: replacement_percentage = {}"
                          .format(self.replacement_percentage))
        self.logger.debug("NEAT NE Algorithm read from config: mutate_prob = {}"
                          .format(self.mutate_prob))
        self.logger.debug("NEAT NE Algorithm read from config: recombine_prob = {}"
                          .format(self.recombine_prob))
        self.logger.debug("NEAT NE Algorithm read from config: mutate_weights_prob = {}"
                          .format(self.mutate_weights_prob))
        self.logger.debug("NEAT NE Algorithm read from config: mutate_connection_prob = {}"
                          .format(self.mutate_connection_prob))
        self.logger.debug("NEAT NE Algorithm read from config: mutate_node_prob = {}"
                          .format(self.mutate_node_prob))
        self.logger.debug("NEAT NE Algorithm read from config: default_activation = {}"
                          .format(self.genome_default_activation))
        self.logger.debug("NEAT NE Algorithm read from config: out_activation = {}"
                          .format(self.genome_out_activation))

        # Check if mutate/recombine and different mutate probabilties are correct set and add up to 1
        assert self.mutate_prob + self.recombine_prob == 1.0
        assert self.mutate_weights_prob + self.mutate_connection_prob + self.mutate_node_prob == 1.0

        # As NEAT evolves model weights manually, set `trainable` to False as automatic weight training should not be
        # possible
        self.trainable = False

    def create_initial_genome(self, input_shape, num_output):
        """
        Create a single direct encoded with a fully connected genotype model, connecting all inputs to all outputs.
        Return this genome.
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
        of the population and replacing them with mutated or recombined genomes, which are based on randomly chosen
        genomes that are left in the population.
        """
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
        raise NotImplementedError("Should implement _create_mutated_genome()")

    def _create_recombined_genome(self, genome_1, genome_2):
        raise NotImplementedError("Should implement _create_recombined_genome()")
