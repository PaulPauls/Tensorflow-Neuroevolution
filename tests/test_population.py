import os
import tensorflow as tf
from absl import logging
from collections import deque

import neuroevolution as ne


class DummyNEAlgorithm():
    def uses_speciation(self):
        return True


def test_population():
    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    config = ne.load_config('./test_config.cfg')

    environment = ne.environments.XOREnvironment()
    environment_name = environment.__class__.__name__
    genome_eval_function = environment.eval_genome_fitness

    encoding = ne.encodings.DirectEncoding(config)
    dummy_ne_algorithm = DummyNEAlgorithm()

    activation_out = tf.keras.activations.deserialize("sigmoid")

    genotype_1 = [
        encoding.create_gene_connection(1, 3),
        encoding.create_gene_connection(2, 3),
        encoding.create_gene_node(3, activation_out)
    ]
    genotype_2 = [
        encoding.create_gene_connection(1, 3),
        encoding.create_gene_connection(1, 4),
        encoding.create_gene_connection(2, 3),
        encoding.create_gene_connection(2, 4),
        encoding.create_gene_node(3, activation_out),
        encoding.create_gene_node(4, activation_out)
    ]

    genome_1 = encoding.create_genome(genotype=genotype_1,
                                      trainable=False,
                                      associated_species=1,
                                      originated_generation=1)
    genome_2 = encoding.create_genome(genotype=genotype_2,
                                      trainable=False,
                                      associated_species=1,
                                      originated_generation=1)

    population = ne.Population(dummy_ne_algorithm, config)

    population.add_genome(1, genome_1)
    population.add_genome(1, genome_2)
    population.add_genome(1, genome_1)
    population.add_genome(1, genome_2)

    population.generation_counter = 0

    population.evaluate(environment_name, genome_eval_function)

    serialization_path = os.path.abspath("test_serialization.json")
    population.save_population(serialization_path)
    population.load_population(encoding, serialization_path)


if __name__ == '__main__':
    test_population()
