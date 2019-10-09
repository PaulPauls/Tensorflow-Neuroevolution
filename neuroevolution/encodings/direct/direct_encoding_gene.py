class DirectEncodingConnection:
    def __init__(self, gene_id, conn_in, conn_out, conn_weight):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out
        self.conn_weight = conn_weight
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled


class DirectEncodingNode:
    def __init__(self, gene_id, node, bias, activation):
        self.gene_id = gene_id
        self.node = node
        self.bias = bias
        self.activation = activation
