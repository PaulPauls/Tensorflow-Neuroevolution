import tensorflow as tf

import neuroevolution as ne
from neuroevolution.environments import CartPoleEnvironment


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()

    logger.debug(tf.__version__)

    env = CartPoleEnvironment()
    config = ne.Config('./yana_example_config.cfg')
    pop = ne.Population()
    ne_algorithm = ne.algorithms.YANA(config, pop)

    engine = ne.EvolutionEngine(ne_algorithm, config, env)

    best_genome = engine.train()
    env.replay_genome(best_genome)


if __name__ == '__main__':
    main()
