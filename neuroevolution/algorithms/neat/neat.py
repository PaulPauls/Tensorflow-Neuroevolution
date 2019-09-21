import tensorflow as tf
from absl import logging

from ..base_algorithm import BaseNeuroevolutionAlgorithm


class NEAT(BaseNeuroevolutionAlgorithm):
    def __init__(self, encoding, config):
        self.encoding = encoding

        # Declare and read in config parameters for the NEAT NE algorithm
        self.genome_elitism = None
        self.recombine_prob = None
        self.mutate_weights_prob = None
        self.add_conn_prob = None
        self.add_node_prob = None
        self.initial_connection = None
        self.species_elitism = None
        self.species_min_size = None
        self.species_max_size = None
        self.species_max_stagnation = None
        self.species_max_fallback = None
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
        self.recombine_prob = config.getfloat(section_name_algorithm, 'recombine_prob')
        self.mutate_weights_prob = config.getfloat(section_name_algorithm, 'mutate_weights_prob')
        self.add_conn_prob = config.getfloat(section_name_algorithm, 'add_conn_prob')
        self.add_node_prob = config.getfloat(section_name_algorithm, 'add_node_prob')
        self.initial_connection = config.get(section_name_algorithm, 'initial_connection')
        self.species_elitism = config.getint(section_name_algorithm, 'species_elitism')
        self.species_min_size = config.getint(section_name_algorithm, 'species_min_size')
        self.species_max_size = config.getint(section_name_algorithm, 'species_max_size')
        self.species_max_stagnation = config.getint(section_name_algorithm, 'species_max_stagnation')
        self.species_max_fallback = config.getfloat(section_name_algorithm, 'species_max_fallback')
        self.species_clustering = config.get(section_name_algorithm, 'species_clustering')
        self.species_interbreeding = config.getboolean(section_name_algorithm, 'species_interbreeding')
        self.activation_default = config.get(section_name_evolvable_encoding, 'activation_default')
        self.activation_out = config.get(section_name_evolvable_encoding, 'activation_out')

        if ',' in self.species_clustering:
            species_clustering_alg = self.species_clustering[:self.species_clustering.find(',')]
            species_clustering_val = float(self.species_clustering[self.species_clustering.find(',') + 2:])
            self.species_clustering = (species_clustering_alg, species_clustering_val)

        self.activation_default = tf.keras.activations.deserialize(self.activation_default)
        self.activation_out = tf.keras.activations.deserialize(self.activation_out)

        logging.debug("NEAT NE Algorithm read from config: genome_elitism = {}".format(self.genome_elitism))
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
        logging.debug("NEAT NE Algorithm read from config: species_max_fallback = {}".format(self.species_max_fallback))
        logging.debug("NEAT NE Algorithm read from config: species_clustering = {}".format(self.species_clustering))
        logging.debug("NEAT NE Algorithm read from config: species_interbreeding = {}"
                      .format(self.species_interbreeding))
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
