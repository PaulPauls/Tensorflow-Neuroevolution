import gym
import logging
import tensorflow as tf
import neuroevolution

from neuroevolution.environments import BaseEnvironment


class Environment(BaseEnvironment):
    """
    ToDo: Implement possibility that an algorithm will require multiple test environments either due to parallel
          execution (see batch_size) or the inability of the environment to be properly reset when testing a new genome.
          Therefore possibly put the creation of this class in the evolution_engine as it will know the batch_size.
    """
    def __init__(self):
        """
        ToDo
        """
        self.env = gym.make('CartPole-v1')

    def eval_genome_fitness(self, genome):
        """
        ToDo: Input genome; apply the genome to the test environments; Return its calculated resulting fitness value
        :param genome:
        :return:
        """
        pass

    def replay_genome(self, genome):
        """
        ToDo: Input genome, apply it to the test environment, though this time render the process of it being applied
        :param genome:
        :return: None
        """
        pass


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Check_1")

    print(tf.__version__)

    env = Environment()

    config = neuroevolution.Config(neuroevolution.algorithms.YANA, 'Path/to/config')

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
