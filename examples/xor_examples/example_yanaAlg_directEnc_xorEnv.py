import tensorflow as tf

import neuroevolution as ne


def main():
    """
    A simple example used in the current alpha stage of development to show of the Tensorflow-Neuroevolution framework.
    This example uses the YANA ne-algorithm with a direct encoded genome to solve the basic XOR environment.

    :return: None
    """

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()
    logger.debug(tf.__version__)

    config = ne.load_config('./example_yanaAlg_directEnc_xorEnv.cfg')

    environment = ne.environments.XOREnvironment(config)

    encoding = ne.encodings.DirectEncoding(config)
    ne_algorithm = ne.algorithms.YANA(config)

    population = ne.Population(encoding, ne_algorithm)

    engine = ne.EvolutionEngine(population, environment, config)

    best_genome = engine.train()
    # best_genome.summary()


if __name__ == '__main__':
    main()
