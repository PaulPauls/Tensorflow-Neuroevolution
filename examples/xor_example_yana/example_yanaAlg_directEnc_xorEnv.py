import tensorflow as tf

import neuroevolution as ne


def main():
    """
    A simple example used in the current alpha stage of development to show of the Tensorflow-Neuroevolution framework.
    This example uses the YANA ne-algorithm with a direct encoded genome to solve the basic XOR environment.

    :return: None
    """
    # Assert that TF 2.x is used
    assert tf.__version__[0] == "2"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logger = tf.get_logger()
    logger.debug(tf.__version__)

    config = ne.load_config('./example_yanaAlg_directEnc_xorEnv.cfg')

    environment = ne.environments.XOREnvironment(config)

    encoding = ne.encodings.DirectEncoding(config)
    ne_algorithm = ne.algorithms.YANA(encoding, config)

    population = ne.Population(ne_algorithm, config)

    engine = ne.EvolutionEngine(population, environment, config)

    best_genome = engine.train(render_best_genome_each_gen=True)

    if best_genome is not None:
        environment.replay_genome(best_genome)
        best_genome.summary()
        best_genome.visualize()
    else:
        logger.info("Evolution of population did not return a valid genome")


if __name__ == '__main__':
    main()
