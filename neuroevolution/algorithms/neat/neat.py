import tensorflow as tf

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class NEAT(BaseNeuroevolutionAlgorithm):
    """
    ToDo: Class docstring
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
        raise NotImplementedError("Should implement create_new_generation()")
