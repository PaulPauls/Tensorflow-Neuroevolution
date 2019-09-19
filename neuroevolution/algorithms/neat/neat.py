from absl import logging

from ..base_algorithm import BaseNeuroevolutionAlgorithm


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

    def initialize_population(self, population, initial_pop_size, input_shape, num_output):
        raise NotImplementedError()

    def evolve_population(self, population):
        raise NotImplementedError()

    def speciate_population(self, population):
        raise NotImplementedError()

    def uses_speciation(self):
        return True
