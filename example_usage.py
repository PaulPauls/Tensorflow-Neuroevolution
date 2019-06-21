import tensorflow as tf

import neuroevolution
from neuroevolution.environments.opengym.cartpole_environment import CartPoleEnvironment


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()

    logger.debug(tf.__version__)

    env = CartPoleEnvironment()
    config = neuroevolution.Config(neuroevolution.algorithms.YANA, './yana_example_config.cfg')
    pop = neuroevolution.Population(config)

    engine = neuroevolution.EvolutionEngine(config, pop, env)

    best_genome = engine.train(max_generations=100)
    env.replay_genome(best_genome)


if __name__ == '__main__':
    main()
