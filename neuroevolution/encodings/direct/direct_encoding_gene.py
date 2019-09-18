import tensorflow as tf


class DirectEncodingConnection:
    def __init__(self, gene_id, conn_in, conn_out, conn_weight):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out
        self.conn_weight = conn_weight

    def serialize(self):
        serialized_gene = {
            'gene_encoding': 'DirectEncodingConnection',
            'gene_id': self.gene_id,
            'conn_in': self.conn_in,
            'conn_out': self.conn_out,
            'conn_weight': str(self.conn_weight)
        }
        return serialized_gene


class DirectEncodingNode:
    def __init__(self, gene_id, node, bias, activation):
        self.gene_id = gene_id
        self.node = node
        self.bias = bias
        self.activation = activation

    def serialize(self):
        serialized_gene = {
            'gene_encoding': 'DirectEncodingNode',
            'gene_id': self.gene_id,
            'node': self.node,
            'bias': str(self.bias),
            'activation': tf.keras.activations.serialize(self.activation)
        }
        return serialized_gene
