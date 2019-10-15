import numpy as np
import tensorflow as tf
from absl import logging

import neuroevolution as ne


def test_direct_encoding():
    """
    Basic test of the TFNE encoding 'DirectEncoding', creating a decently complex genotype, creating a genome from it
    and predicting the results of a XOR function with it.
    """

    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    encoding = ne.encodings.DirectEncoding(dtype=tf.float32, run_eagerly=False)

    activation_default = tf.keras.activations.deserialize("tanh")
    activation_out = tf.keras.activations.deserialize("sigmoid")

    gene_list = list()
    gene_list.append(encoding.create_gene_connection(6, 8, 0.5))
    gene_list.append(encoding.create_gene_connection(5, 7, 0.5))
    gene_list.append(encoding.create_gene_connection(4, 7, 0.5))
    gene_list.append(encoding.create_gene_connection(3, 7, 0.5))
    gene_list.append(encoding.create_gene_connection(3, 8, 0.5))
    gene_list.append(encoding.create_gene_connection(2, 4, 0.513241234))
    gene_list.append(encoding.create_gene_connection(1, 3, 0.5))
    gene_list.append(encoding.create_gene_connection(1, 4, 0.5))
    gene_list.append(encoding.create_gene_connection(2, 5, 0.5))
    gene_list.append(encoding.create_gene_connection(3, 6, 0.5))
    gene_list.append(encoding.create_gene_connection(4, 6, 0.5))
    gene_list.append(encoding.create_gene_connection(5, 6, 0.5))
    gene_list.append(encoding.create_gene_connection(5, 8, 0.5))
    gene_list.append(encoding.create_gene_connection(7, 8, 0.5))
    gene_list.append(encoding.create_gene_node(6, 0, activation_default))
    gene_list.append(encoding.create_gene_node(4, 0, activation_default))
    gene_list.append(encoding.create_gene_node(3, 0, activation_default))
    gene_list.append(encoding.create_gene_node(5, 0, activation_default))
    gene_list.append(encoding.create_gene_node(7, 0, activation_default))
    gene_list.append(encoding.create_gene_node(8, 0, activation_out))

    genotype = dict()
    for (gene_id, gene) in gene_list:
        genotype[gene_id] = gene

    genome_id, genome = encoding.create_genome(genotype=genotype)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = genome.get_model()

    print(model.predict(x))
    genome.visualize()


if __name__ == '__main__':
    test_direct_encoding()
