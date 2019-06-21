import logging
import tensorflow as tf

import neuroevolution

from neuroevolution.environments.opengym.cartpole_environment import CartPoleEnvironment


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Check_1")

    print(tf.__version__)

    env = CartPoleEnvironment()

    config = neuroevolution.Config(neuroevolution.algorithms.YANA, './yana_example_config.cfg')

    pop = neuroevolution.Population(config)

    engine = neuroevolution.EvolutionEngine(config, pop, env.eval_genome_fitness)

    engine.set_verbosity(5)
    # Alternatively: engine.add_reporter(StdOutReporter(True))
    # Alternatively: Proper TF way of adding reporters/verbosity

    best_genome = engine.train(max_generations=100)
    # Alternatively:
    #   engine.train()
    #   best_genome = pop.get_best_genome()

    env.replay_genome(best_genome)
