import tensorflow as tf
from absl import logging

from ..base_encoding import BaseEncoding
from .direct_encoding_genome import DirectEncodingGenome
from .direct_encoding_gene import DirectEncodingConnection, DirectEncodingNode


class DirectEncoding(BaseEncoding):
    def __init__(self, config):
        self.gene_id_counter = 0
        self.genome_id_counter = 0
        self.gene_to_gene_id_mapping = dict()

        # Declare and read in config parameters for the Direct encoding
        self.initializer_kernel = None
        self.initializer_bias = None
        self.dtype = None
        self._read_config_parameters(config)

    def _read_config_parameters(self, config):
        section_name = 'DIRECT_ENCODING' if config.has_section('DIRECT_ENCODING') else 'ENCODING'
        self.initializer_kernel = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_kernel'))
        self.initializer_bias = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_bias'))
        self.dtype = tf.dtypes.as_dtype(config.get(section_name, 'dtype'))

        logging.debug("Direct Encoding read from config: initializer_kernel = {}".format(self.initializer_kernel))
        logging.debug("Direct Encoding read from config: initializer_bias = {}".format(self.initializer_bias))
        logging.debug("Direct Encoding read from config: dtype = {}".format(self.dtype))

    def create_gene_connection(self, conn_in, conn_out):
        # Determine unique gene_id by checking if the supplied (conn_in, conn_out) pair already created a gene.
        # If not, increment gene_id_counter and register this new unique gene_id to the supplied connection pair.
        gene_key = frozenset((conn_in, conn_out))
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            gene_id = self.gene_id_counter
            self.gene_to_gene_id_mapping[gene_key] = gene_id

        # Create an initial connection weight by using the supplied kernel initializer (which determines the initial
        # connection weights) to create a single random value
        conn_weight = tf.Variable(initial_value=self.initializer_kernel((1,)), dtype=self.dtype, shape=(1,)).numpy()[0]

        return DirectEncodingConnection(gene_id, conn_in, conn_out, conn_weight)

    def create_gene_node(self, node, activation):
        # Determine unique gene_id by checking if the supplied node already created a gene.
        # If not, increment gene_id_counter and register this new unique gene_id to the supplied node.
        gene_key = frozenset((node,))
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            gene_id = self.gene_id_counter
            self.gene_to_gene_id_mapping[gene_key] = gene_id

        # Create a bias value by using the supplied bias initializer to create a single random value
        bias = tf.Variable(initial_value=self.initializer_bias((1,)), dtype=self.dtype, shape=(1,)).numpy()[0]

        return DirectEncodingNode(gene_id, node, bias, activation)

    def create_genome(self, genotype, trainable):
        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype, trainable, self.dtype)
