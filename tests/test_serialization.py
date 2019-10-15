import os
import tensorflow as tf
from copy import deepcopy
from absl import logging

import neuroevolution as ne


def test_population():
    """
    TODO DOC
    """

    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    # Create the required NE faculties, among them being the population to serialize and the encoding to create genomes
    # with which the population will be populated
    config = ne.load_config('./test_config.cfg')
    encoding = ne.encodings.DirectEncoding(trainable=False, dtype=tf.float32, run_eagerly=False)
    population = ne.Population(config, None)

    activation_output = tf.keras.activations.deserialize("sigmoid")

    # Create 2 Seperate genotypes and clone them for a total of 4 genotypes
    gene_list_1 = list()
    gene_list_1.append(encoding.create_gene_connection(1, 3, 0.4))
    gene_list_1.append(encoding.create_gene_connection(2, 3, 0.6))
    gene_list_1.append((encoding.create_gene_node(3, 0.8, activation_output)))
    genotype_1 = dict()
    for (gene_id, gene) in gene_list_1:
        genotype_1[gene_id] = gene

    gene_list_2 = list()
    gene_list_2.append(encoding.create_gene_connection(1, 3, 0.2))
    gene_list_2.append(encoding.create_gene_connection(1, 4, 0.3))
    gene_list_2.append(encoding.create_gene_connection(2, 3, 0.4))
    gene_list_2.append(encoding.create_gene_connection(2, 4, 0.5))
    gene_list_2.append((encoding.create_gene_node(3, 0.6, activation_output)))
    gene_list_2.append((encoding.create_gene_node(4, 0.7, activation_output)))
    genotype_2 = dict()
    for (gene_id, gene) in gene_list_2:
        genotype_2[gene_id] = gene

    genotype_3 = deepcopy(genotype_1)
    genotype_4 = deepcopy(genotype_2)

    # Create genomes from genotypes and set their fitness to dummy values
    genome_1_id, genome_1 = encoding.create_genome(genotype_1)
    genome_1.set_fitness(12)
    genome_2_id, genome_2 = encoding.create_genome(genotype_2)
    genome_2.set_fitness(34)
    genome_3_id, genome_3 = encoding.create_genome(genotype_3)
    genome_3.set_fitness(56)
    genome_4_id, genome_4 = encoding.create_genome(genotype_4)
    genome_4.set_fitness(78)

    # Populate the population with created genomes
    population.generation_counter = 0
    population.add_genome(genome_1_id, genome_1)
    population.add_genome(genome_2_id, genome_2)
    population.add_genome(genome_3_id, genome_3)
    population.add_genome(genome_4_id, genome_4)

    # Save population to file 'test_serialization' in current working directory
    serialization_path = os.path.abspath("test_serialization.json")
    population.save_population(serialization_path)


if __name__ == '__main__':
    test_population()
