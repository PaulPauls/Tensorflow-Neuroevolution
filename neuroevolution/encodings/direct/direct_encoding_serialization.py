import tensorflow as tf


def deserialize_genome_list(genome_list):
    raise NotImplementedError()


def serialize_genome(genome):
    serialized_genome = {
        'genome_encoding': 'DirectEncodingGenome',
        'genome_id': genome.genome_id,
        'fitness': genome.fitness,
        'trainable': genome.trainable,
        'dtype': str(genome.dtype),
        'run_eagerly': genome.run_eagerly,
        'genotype': [gene.serialize() for gene in genome.genotype]
    }
    return serialized_genome


def serialize_connection_gene(connection_gene):
    serialized_gene = {
        'gene_encoding': 'DirectEncodingConnection',
        'gene_id': connection_gene.gene_id,
        'conn_in': connection_gene.conn_in,
        'conn_out': connection_gene.conn_out,
        'conn_weight': float(connection_gene.conn_weight)
    }
    return serialized_gene


def serialize_node_gene(node_gene):
    serialized_gene = {
        'gene_encoding': 'DirectEncodingNode',
        'gene_id': node_gene.gene_id,
        'node': node_gene.node,
        'bias': float(node_gene.bias),
        'activation': tf.keras.activations.serialize(node_gene.activation)
    }
    return serialized_gene
