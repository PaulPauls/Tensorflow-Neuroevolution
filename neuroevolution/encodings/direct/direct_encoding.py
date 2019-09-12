import tensorflow as tf
from collections import deque
from absl import logging

from neuroevolution.encodings import BaseEncoding
from .direct_encoding_genome import DirectEncodingGenome
from .direct_encoding_gene import DirectEncodingGene, DirectEncodingGeneIDBank


def deserialize_genome(genotype, activations):
    deserialized_genotype = deque()

    if isinstance(genotype, list):
        gene_id_bank = DirectEncodingGeneIDBank()
        for (conn_in, conn_out) in genotype:
            new_gene = DirectEncodingGene(gene_id_bank.get_id((conn_in, conn_out)), conn_in, conn_out)
            deserialized_genotype.append(new_gene)
    elif isinstance(genotype, dict):
        for gene_id, conns in genotype.items():
            new_gene = DirectEncodingGene(gene_id, conns[0], conns[1])
            deserialized_genotype.append(new_gene)


    # Convert activation functions to the actual tensorflow functions if they are supplied as strings
    if isinstance(activations['out_activation'], str):
        activations['out_activation'] = tf.keras.activations.deserialize(activations['out_activation'])
    if isinstance(activations['default_activation'], str):
        activations['default_activation'] = tf.keras.activations.deserialize(activations['default_activation'])
    if len(activations.keys()) > 2:
        raise NotImplementedError("activation dict contains more activations than the 'out' and 'default' activation")

    return deserialized_genotype, activations


def check_genome_sanity_function(genotype, activations):
    pass


class DirectEncoding(BaseEncoding):
    def __init__(self, config):
        self.genome_id_counter = 0

        # Declare and read in config parameters for the Direct encoding
        self.check_genome_sanity = None
        self.initializer_kernel = None
        self.initializer_bias = None
        self.dtype = None
        self._read_config_parameters(config)

    def _read_config_parameters(self, config):
        section_name = 'DIRECT_ENCODING' if config.has_section('DIRECT_ENCODING') else 'ENCODING'
        self.check_genome_sanity = config.getboolean(section_name, 'check_genome_sanity')
        self.initializer_kernel = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_kernel'))
        self.initializer_bias = tf.keras.initializers.deserialize(config.get(section_name, 'initializer_bias'))
        self.dtype = tf.dtypes.as_dtype(config.get(section_name, 'dtype'))

        logging.debug("Direct Encoding read from config: check_genome_sanity = {}".format(self.check_genome_sanity))
        logging.debug("Direct Encoding read from config: initializer_kernel = {}".format(self.initializer_kernel))
        logging.debug("Direct Encoding read from config: initializer_bias = {}".format(self.initializer_bias))
        logging.debug("Direct Encoding read from config: dtype = {}".format(self.dtype))

    def create_new_genome(self, genotype, activations, trainable, check_genome_sanity=None):
        check_genome_sanity = self.check_genome_sanity if check_genome_sanity is None else check_genome_sanity

        # ToDo: doc
        if isinstance(genotype, list) or isinstance(genotype, dict):
            genotype, activations = deserialize_genome(genotype, activations)

        if check_genome_sanity:
            check_genome_sanity_function(genotype, activations)

        self.genome_id_counter += 1
        return DirectEncodingGenome(genome_id=self.genome_id_counter,
                                    genotype=genotype,
                                    activations=activations,
                                    initializer_kernel=self.initializer_kernel,
                                    initializer_bias=self.initializer_bias,
                                    trainable=trainable,
                                    dtype=self.dtype)
