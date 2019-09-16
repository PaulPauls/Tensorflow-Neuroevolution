import tensorflow as tf
from toposort import toposort

from .direct_encoding_gene import DirectEncodingConnection, DirectEncodingNode


class DirectEncodingModel(tf.keras.Model):
    def __init__(self, genotype, trainable, dtype, run_eagerly):
        super(DirectEncodingModel, self).__init__(trainable=trainable, dtype=dtype)
        self.run_eagerly = run_eagerly

        nodes, connection, node_dependencies = self._create_gene_dicts(genotype)

        


    @staticmethod
    def _create_gene_dicts(genotype):
        nodes = dict()
        connections = dict()
        node_dependencies = dict()

        for gene in genotype:
            if isinstance(gene, DirectEncodingConnection):
                conn_out = gene.conn_out
                conn_in = gene.conn_in
                if conn_out in connections:
                    connections[conn_out].append((conn_in, gene.conn_weight))
                    node_dependencies[conn_out].add(conn_in)
                else:
                    connections[conn_out] = [(conn_in, gene.conn_weight)]
                    node_dependencies[conn_out] = {conn_in}

            else:  # else gene is instance of DirectEncodingNode
                node = gene.node
                if node in nodes:
                    nodes[node].append((gene.bias, gene.activation))
                else:
                    nodes[node] = [(gene.bias, gene.activation)]

        return nodes, connections, node_dependencies

    def call(self, inputs):
        raise NotImplementedError()
