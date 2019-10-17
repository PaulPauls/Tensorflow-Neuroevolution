import numpy as np
import tensorflow as tf
from absl import logging

import neuroevolution as ne


def test_cartpole_environment():
    """
    Basic test of the TFNE CartPole environment implementation, representing an interface to OpenAI Gym's Cartpole
    environment. The test uses a non-trainable DirectEncoding and initializes it with the minimal neural network, having
    no hidden nodes and all inputs connected to all outputs.
    """

    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    environment = ne.environments.CartPoleEnvironment(render_flag=False)
    encoding = ne.encodings.DirectEncoding(trainable=False, dtype=tf.float32)

    activation_output = tf.keras.activations.deserialize("sigmoid")
    num_input = environment.get_input_shape()[0]
    num_output = environment.get_num_output()
    genotype = dict()
    for conn_in in range(1, num_input + 1):
        for conn_out in range(num_input + 1, num_input + num_output + 1):
            conn_weight = np.random.normal(loc=0.5, scale=0.5)
            gene_id, gene_conn = encoding.create_gene_connection(conn_in, conn_out, conn_weight)
            genotype[gene_id] = gene_conn
    for node in range(num_input + 1, num_input + num_output + 1):
        gene_id, gene_node = encoding.create_gene_node(node, 0, activation_output)
        genotype[gene_id] = gene_node

    _, genome = encoding.create_genome(genotype)

    fitness = environment.eval_genome_fitness(genome)
    print("Evaluated Fitness: {}".format(fitness))
    environment.replay_genome(genome)
    genome.visualize()


if __name__ == '__main__':
    test_cartpole_environment()
