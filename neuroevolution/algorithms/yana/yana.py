from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class YANA(BaseNeuroevolutionAlgorithm):
    """
    Test implementation of the the dummy 'Yet Another Neuroevolution Algorithm', which does all required Neuroevolution
    Algorithm tasks in the most basic way to enable testing the framework.
    """
    def __init__(self, encoding, config):
        self.encoding = encoding

        # Read in config parameters for neuroevolution algorithm
        self.genome_default_activation = config.get('NE_ALGORITHM', 'default_activation')
        self.genome_out_activation = config.get('NE_ALGORITHM', 'out_activation')

    def create_initial_genome(self, input_shape, num_output):
        # Create as initial genome a fully connected (for now) phenotype with specified number of inputs and outputs
        genotype = dict()
        trainable = True

        # Determine if multidimensional input vector (as this is not yet implemented
        if len(input_shape) == 1:
            num_input = input_shape[0]

            # Create a connection from each input node to each output node
            key_counter = 1
            for in_node in range(1, num_input+1):
                for out_node in range(num_input+1, num_input+num_output+1):
                    conn_in_out = (in_node, out_node)
                    genotype[key_counter] = conn_in_out
                    key_counter += 1

            # Specify layer activation functions for genotype
            activations = {'out_activation': self.genome_out_activation,
                           'default_activation': self.genome_default_activation}

        else:
            raise NotImplementedError("Multidimensional Input vector not yet supported")

        new_initialized_genome = self.encoding.create_new_genome(genotype, activations, trainable=trainable)
        return new_initialized_genome

    def create_mutated_genome(self, genome):
        return self.create_initial_genome((2,), 1)
