class DirectEncodingGene:
    def __init__(self, gene_id, conn_in, conn_out):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out


class DirectEncodingGeneIDBank:
    gene_id_counter = 0
    connection_tuple_to_id_mapping = dict()

    def __init__(self):
        pass

    def get_id(self, connection_tuple):
        connection_tuple = frozenset(connection_tuple)
        if connection_tuple in DirectEncodingGeneIDBank.connection_tuple_to_id_mapping:
            return DirectEncodingGeneIDBank.connection_tuple_to_id_mapping[connection_tuple]

        DirectEncodingGeneIDBank.gene_id_counter += 1
        DirectEncodingGeneIDBank.connection_tuple_to_id_mapping[connection_tuple] = \
            DirectEncodingGeneIDBank.gene_id_counter
        return DirectEncodingGeneIDBank.gene_id_counter
