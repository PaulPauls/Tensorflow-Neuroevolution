import tensorflow as tf
from absl import logging
from collections import deque

from ..base_encoding import BaseEncoding
from .direct_encoding_genome import DirectEncodingGenome
from .direct_encoding_gene import DirectEncodingConnection, DirectEncodingNode


class DirectEncoding(BaseEncoding):
    def __init__(self, config, dtype=tf.float32, run_eagerly=False):
        self.gene_id_counter = 0
        self.genome_id_counter = 0
        self.gene_to_gene_id_mapping = dict()

        # Declare and read in config parameters for the Direct encoding
        self.initializer_kernel = None
        self.initializer_bias = None
        self._read_config_parameters(config)
        self.dtype = dtype
        self.run_eagerly = run_eagerly
        self._log_class_parameters()

    def _read_config_parameters(self, config):
        section_name = 'DIRECT_ENCODING' if config.has_section('DIRECT_ENCODING') else 'ENCODING'
        self.initializer_kernel = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_kernel'))
        self.initializer_bias = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_bias'))

        logging.debug("Direct Encoding read from config: initializer_kernel = {}".format(self.initializer_kernel))
        logging.debug("Direct Encoding read from config: initializer_bias = {}".format(self.initializer_bias))

    def _log_class_parameters(self):
        logging.debug("Direct Encoding parameter: dtype = {}".format(self.dtype))
        logging.debug("Direct Encoding parameter: run_eagerly = {}".format(self.run_eagerly))

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
        init_value = self.initializer_kernel(shape=(1,), dtype=self.dtype)
        conn_weight = tf.Variable(initial_value=init_value, dtype=self.dtype, shape=(1,)).numpy()[0]

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
        init_value = self.initializer_bias(shape=(1,), dtype=self.dtype)
        bias = tf.Variable(initial_value=init_value, dtype=self.dtype, shape=(1,)).numpy()[0]

        return DirectEncodingNode(gene_id, node, bias, activation)

    def create_genome(self, genotype, trainable):
        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype, trainable, self.dtype, self.run_eagerly)

    def deserialize_genome_list(self, genome_list):
        deserialized_genome_list = []
        for genome in genome_list:
            genome_id = genome['genome_id']
            fitness = genome['fitness']
            trainable = genome['trainable']
            dtype = tf.dtypes.as_dtype(genome['dtype'])
            run_eagerly = genome['run_eagerly']
            assert dtype == self.dtype
            assert run_eagerly == self.run_eagerly

            genotype = deque([])
            for gene in genome['genotype']:
                if gene['gene_encoding'] == "DirectEncodingConnection":
                    gene_id = gene['gene_id']
                    conn_in = gene['conn_in']
                    conn_out = gene['conn_out']
                    conn_weight = dtype.as_numpy_dtype(gene['conn_weight'])
                    deserialized_gene = DirectEncodingConnection(gene_id, conn_in, conn_out, conn_weight)
                else:
                    gene_id = gene['gene_id']
                    node = gene['node']
                    bias = dtype.as_numpy_dtype(gene['bias'])
                    activation = tf.keras.activations.deserialize(gene['activation'])
                    deserialized_gene = DirectEncodingNode(gene_id, node, bias, activation)
                genotype.append(deserialized_gene)

            deserialized_genome = DirectEncodingGenome(genome_id, genotype, trainable, dtype, run_eagerly)
            deserialized_genome.set_fitness(fitness)

            deserialized_genome_list.append(deserialized_genome)

        return deserialized_genome_list
