import numpy as np
import tensorflow as tf
from absl import logging
from collections import deque

import neuroevolution as ne


def test_direct_encoding():
    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    config = ne.load_config('./test_config.cfg')

    encoding = ne.encodings.DirectEncoding(config, dtype=tf.float32, run_eagerly=False)

    activation_default = tf.keras.activations.deserialize("tanh")
    activation_out = tf.keras.activations.deserialize("sigmoid")

    genotype = [
        encoding.create_gene_connection(6, 8),
        encoding.create_gene_connection(5, 7),
        encoding.create_gene_connection(4, 7),
        encoding.create_gene_connection(3, 7),
        encoding.create_gene_connection(3, 8),
        encoding.create_gene_connection(2, 4),
        encoding.create_gene_connection(1, 3),
        encoding.create_gene_connection(1, 4),
        encoding.create_gene_connection(2, 5),
        encoding.create_gene_connection(3, 6),
        encoding.create_gene_connection(4, 6),
        encoding.create_gene_connection(5, 6),
        encoding.create_gene_connection(5, 8),
        encoding.create_gene_connection(7, 8),
        encoding.create_gene_node(6, activation_default),
        encoding.create_gene_node(4, activation_default),
        encoding.create_gene_node(3, activation_default),
        encoding.create_gene_node(5, activation_default),
        encoding.create_gene_node(7, activation_default),
        encoding.create_gene_node(8, activation_out)
    ]

    genome = encoding.create_genome(genotype=genotype, trainable=False, associated_species=1, originated_generation=1)

    print(genome)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = genome.get_model()

    print(model.predict(x))


if __name__ == '__main__':
    test_direct_encoding()
