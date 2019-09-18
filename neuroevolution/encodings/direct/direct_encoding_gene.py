from .direct_encoding_serialization import serialize_connection_gene, serialize_node_gene


class DirectEncodingConnection:
    def __init__(self, gene_id, conn_in, conn_out, conn_weight):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out
        self.conn_weight = conn_weight

    def serialize(self):
        serialize_connection_gene(self)


class DirectEncodingNode:
    def __init__(self, gene_id, node, bias, activation):
        self.gene_id = gene_id
        self.node = node
        self.bias = bias
        self.activation = activation

    def serialize(self):
        serialize_node_gene(self)
