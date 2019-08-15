import tensorflow as tf
from collections import deque

from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings.direct import DirectEncodingGenome, DirectEncodingGene


def deserialize_genome(genotype, activations):
    # Convert Key of genotype_dict to gene_id and create with specified connection the custom gene class. Then join
    # all genes in a double ended linked list
    deserialized_genotype = deque()
    for gene_id, conns in genotype.items():
        new_gene = DirectEncodingGene(gene_id, conns[0], conns[1])
        deserialized_genotype.append(new_gene)

    # Convert activation functions to the actual tensorflow functions if they are supplied as strings
    if isinstance(activations['out_activation'], str):
        activations['out_activation'] = activation_string_to_function(activations['out_activation'])
    if isinstance(activations['default_activation'], str):
        activations['default_activation'] = activation_string_to_function(activations['default_activation'])
    if len(activations.keys()) > 2:
        raise NotImplementedError("activation dict contains more activations than the 'out' and 'default' activation")

    return deserialized_genotype, activations


def check_genome_sanity_function(genotype, activations):
    raise NotImplementedError()


def activation_string_to_function(activation_string):
    if activation_string == "softmax":
        return tf.keras.activations.softmax
    if activation_string == "sigmoid":
        return tf.keras.activations.sigmoid
    if activation_string == "tanh":
        return tf.keras.activations.tanh
    raise NotImplementedError("requested activation function not yet implemented in tensorflow function lookup")


class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        self.genome_id_counter = 0

        # Read in config parameters for the genome encoding
        self.check_genome_sanity = config.getboolean('ENCODING', 'check_genome_sanity')

    def create_new_genome(self, genotype, activations, trainable=True, check_genome_sanity=None):
        check_genome_sanity = self.check_genome_sanity if check_genome_sanity is None else check_genome_sanity

        # if genotype is explicitely specified in dict form it will be deserialized into a linked-List. Otherwise
        # we assume that the genotype is in the correct form (deque of DirectEncodingGenes). If this is not the
        # case will check_genome_sanity() throw an error
        if isinstance(genotype, dict):
            genotype, activations = deserialize_genome(genotype, activations)

        if check_genome_sanity:
            check_genome_sanity_function(genotype, activations)

        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype, activations, trainable=trainable)
