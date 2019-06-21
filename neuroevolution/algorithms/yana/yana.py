import tensorflow as tf

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

        pass

    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        pass

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
