import tensorflow as tf

import neuroevolution as ne


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()

    logger.debug(tf.__version__)

    config = ne.Config('./fashion_mnist_example_config.cfg')
    env = ne.environments.FashionMNISTEnvironment(config)
    pop = ne.Population()
    exit(1)

    ne_algorithm = ne.algorithms.YANA(config, pop)

    engine = ne.EvolutionEngine(ne_algorithm, config, env)

    best_genome = engine.train(max_generations=2)
    env.replay_genome(best_genome)


if __name__ == '__main__':
    main()
