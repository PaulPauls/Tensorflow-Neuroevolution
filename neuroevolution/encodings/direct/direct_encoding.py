import tensorflow as tf
from absl import logging

from ..base_encoding import BaseEncoding
from .direct_encoding_genome import DirectEncodingGenome
from .direct_encoding_gene import DirectEncodingConnection, DirectEncodingNode


class DirectEncoding(BaseEncoding):
    def __init__(self, config, dtype=tf.float32, run_eagerly=False):
        self.dtype = dtype
        self.run_eagerly = run_eagerly
        self.initializer_kernel = None
        self.initializer_bias = None
        self._read_config_parameters(config)
        self._log_class_parameters()

        self.gene_id_counter = 0
        self.genome_id_counter = 0
        self.gene_to_gene_id_mapping = dict()

    def _read_config_parameters(self, config):
        section_name = 'DIRECT_ENCODING' if config.has_section('DIRECT_ENCODING') else 'ENCODING'
        self.initializer_kernel = config.get(section_name, 'initializer_kernel')
        self.initializer_bias = config.get(section_name, 'initializer_bias')

        self.initializer_kernel = tf.keras.initializers.deserialize(self.initializer_kernel)
        self.initializer_bias = tf.keras.initializers.deserialize(self.initializer_bias)

    def _log_class_parameters(self):
        logging.debug("Direct Encoding parameter: dtype = {}".format(self.dtype))
        logging.debug("Direct Encoding parameter: run_eagerly = {}".format(self.run_eagerly))
        logging.debug("Direct Encoding read from config: initializer_kernel = {}".format(self.initializer_kernel))
        logging.debug("Direct Encoding read from config: initializer_bias = {}".format(self.initializer_bias))

    def create_gene_connection(self, conn_in, conn_out):
        # Determine unique gene_id by checking if the supplied (conn_in, conn_out) pair already created a gene.
        # If not, increment gene_id_counter and register this new unique gene_id to the supplied connection pair.
        gene_key = (conn_in, conn_out)
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            gene_id = self.gene_id_counter
            self.gene_to_gene_id_mapping[gene_key] = gene_id

        # Create an initial connection weight by using the supplied kernel initializer (which determines the initial
        # connection weights) to create a single random value
        init_value = self.initializer_kernel(shape=(1,), dtype=self.dtype)
        conn_weight = tf.Variable(initial_value=init_value, dtype=self.dtype, shape=(1,)).numpy()[0]

        return DirectEncodingConnection(gene_id, conn_in, conn_out, conn_weight)

    def create_gene_node(self, node, activation):
        # Determine unique gene_id by checking if the supplied node already created a gene.
        # If not, increment gene_id_counter and register this new unique gene_id to the supplied node.
        gene_key = (node,)
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            gene_id = self.gene_id_counter
            self.gene_to_gene_id_mapping[gene_key] = gene_id

        # Create a bias value by using the supplied bias initializer to create a single random value
        init_value = self.initializer_bias(shape=(1,), dtype=self.dtype)
        bias = tf.Variable(initial_value=init_value, dtype=self.dtype, shape=(1,)).numpy()[0]

        return DirectEncodingNode(gene_id, node, bias, activation)

    def create_genome(self, genotype, trainable):
        self.genome_id_counter += 1
        return DirectEncodingGenome(genome_id=self.genome_id_counter,
                                    genotype=genotype,
                                    trainable=trainable,
                                    dtype=self.dtype,
                                    run_eagerly=self.run_eagerly)
