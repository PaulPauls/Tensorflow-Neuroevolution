import os
import tensorflow as tf
from absl import logging
from collections import deque

import neuroevolution as ne


def test_population():
    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    config = ne.load_config('./test_config.cfg')

    encoding = ne.encodings.DirectEncoding(config)

    activation_out = tf.keras.activations.deserialize("sigmoid")

    genotype_1 = deque([
        encoding.create_gene_connection(1, 3),
        encoding.create_gene_connection(2, 3),
        encoding.create_gene_node(3, activation_out)
    ])
    genotype_2 = deque([
        encoding.create_gene_connection(1, 3),
        encoding.create_gene_connection(1, 4),
        encoding.create_gene_connection(2, 3),
        encoding.create_gene_connection(2, 4),
        encoding.create_gene_node(3, activation_out),
        encoding.create_gene_node(4, activation_out)
    ])

    genome_1 = encoding.create_genome(genotype_1, trainable=False)
    genome_2 = encoding.create_genome(genotype_2, trainable=False)

    population = ne.Population(None, config)

    population.append_genome(genome_1)
    population.append_genome(genome_2)
    population.append_genome(genome_1)
    population.append_genome(genome_2)

    population.generation_counter = 0

    serialization_path = os.path.abspath("test_serialization.json")
    population.save_population(serialization_path)
    population.load_population(encoding, serialization_path)


if __name__ == '__main__':
    test_population()
