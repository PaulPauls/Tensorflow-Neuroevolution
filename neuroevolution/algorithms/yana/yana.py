import tensorflow as tf

from neuroevolution.genome import Genome
from neuroevolution.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm


class YANA(BaseNeuroevolutionAlgorithm):
    """
    ToDo: Test implementation of the the dummy 'Yet Another Neuroevolution Algorithm', which does all required
          Neuroevolution Algorithm tasks in the most basic way to enable testing the framework.
    """
    def __init__(self, config, population):
        """
        ToDo
        :param config:
        :param population:
        """
        self.logger = tf.get_logger()

        self.config = config
        self.population = population

    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        pop_size = int(self.config.algorithm_parameters['YANA']['pop_size'])
        self.population.create_initial_population(pop_size, self.create_genome)

    @staticmethod
    def create_genome():
        """
        ToDo
        :return:
        """
        return Genome()

    def select_genomes(self):
        """
        ToDo
        :return:
        """
        pass

    def recombine_genomes(self):
        """
        ToDo
        :return:
        """
        pass

    def mutate_genomes(self):
        """
        ToDo
        :return:
        """
        pass
