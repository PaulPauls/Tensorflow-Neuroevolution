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
        raise NotImplementedError("Should implement create_initial_genome()")

    def create_new_generation(self, population):
        raise NotImplementedError("Should implement create_new_generation()")
