import tensorflow as tf

import neuroevolution as ne


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()

    logger.debug(tf.__version__)

    config = ne.load_config('./fashion_mnist_example_config.cfg')
    env = ne.environments.FashionMNISTEnvironment(config)
    pop = ne.Population()

    input_shape = env.get_input_shape()
    num_output = env.get_num_output()
    encoding = ne.encodings.KerasLayerEncoding(input_shape, num_output, config)
    ne_algorithm = ne.algorithms.YANA(encoding, pop, config)

    exit(13)

    engine = ne.EvolutionEngine(ne_algorithm, pop, env, config)



    best_genome = engine.train(max_generations=2)
    env.replay_genome(best_genome)


if __name__ == '__main__':
    main()
