import tensorflow as tf
from absl import logging

from ..base_encoding import BaseEncoding
from .direct_encoding_genome import DirectEncodingGenome
from .direct_encoding_gene import DirectEncodingConnection, DirectEncodingNode


class DirectEncoding(BaseEncoding):
    """
    Factory Wrapper for DirectEncoding genomes, providing unique continuous gene- and genome-ids for created genes and
    genomes as well as genomes created with the supplied parameters for trainable, dtype and run_eagerly
    """

    def __init__(self, trainable=False, dtype=tf.float32, run_eagerly=False):
        self.trainable = trainable
        self.dtype = dtype
        self.run_eagerly = run_eagerly
        self._log_class_parameters()

        self.gene_id_counter = 0
        self.genome_id_counter = 0
        self.gene_to_gene_id_mapping = dict()

    def _log_class_parameters(self):
        logging.debug("Direct Encoding parameter: trainable = {}".format(self.trainable))
        logging.debug("Direct Encoding parameter: dtype = {}".format(self.dtype))
        logging.debug("Direct Encoding parameter: run_eagerly = {}".format(self.run_eagerly))

    def create_gene_connection(self, conn_in, conn_out, conn_weight) -> (int, DirectEncodingConnection):
        """
        Create DirectEncoding connection gene with unique continuous gene-id based on the supplied (conn_in, conn_out)
        tuple. Uniqueness disregards conn_weight, meaning that identical gene_ids with different conn_weights can exist.
        :param conn_in: node (usually int) the connection is originating from
        :param conn_out: node (usually int) the connection is ending in
        :param conn_weight: weight (usually float or np.float) of the connection
        :return: tuple of unique gene-id and created DirectEncoding connection gene
        """
        gene_key = (conn_in, conn_out)
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            self.gene_to_gene_id_mapping[gene_key] = self.gene_id_counter
            gene_id = self.gene_id_counter

        return gene_id, DirectEncodingConnection(gene_id, conn_in, conn_out, conn_weight)

    def create_gene_node(self, node, bias, activation) -> (int, DirectEncodingNode):
        """
        Create DirectEncoding node gene with unique continuous gene-id based on the supplied node. Uniqueness disregards
        bias and activation, meaning that identical gene_ids with different bias and activation can exist.
        :param node: node (usually int) the gene represents
        :param bias: bias weight (usually float or np.float) of the node
        :param activation: Tensorflow activation function of the node
        :return: tuple of unique gene-id and created DirectEncoding node gene
        """
        gene_key = (node,)
        if gene_key in self.gene_to_gene_id_mapping:
            gene_id = self.gene_to_gene_id_mapping[gene_key]
        else:
            self.gene_id_counter += 1
            self.gene_to_gene_id_mapping[gene_key] = self.gene_id_counter
            gene_id = self.gene_id_counter

        return gene_id, DirectEncodingNode(gene_id, node, bias, activation)

    def create_genome(self, genotype) -> (int, DirectEncodingGenome):
        """
        Create DirectEncoding genome with continuous genome-id for each newly created genome
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :return: tuple of continuous genome-id and created DirectEncoding genome
        """
        self.genome_id_counter += 1
        return self.genome_id_counter, DirectEncodingGenome(genome_id=self.genome_id_counter,
                                                            genotype=genotype,
                                                            trainable=self.trainable,
                                                            dtype=self.dtype,
                                                            run_eagerly=self.run_eagerly)
