import tensorflow as tf

from neuroevolution.encodings import KerasLayerEncodingGenome
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
        # Somehow determine the input_shape and num_output for environment. For now I just magically know them
        input_shape = (28, 28)
        num_output = 10
        return KerasLayerEncodingGenome(input_shape, num_output)

    def select_genomes(self):
        """
        ToDo
        :return:
        """
        # for now, delete the 20% of the population that is performing the lowest
        pop_size = int(self.config.algorithm_parameters['YANA']['pop_size'])
        for _ in range(int(pop_size * 0.2)):
            worst_genome = min(self.population.genome_list, key=lambda x: x.fitness)
            self.logger.debug("Genome with fitness {} deleted".format(worst_genome.fitness))
            self.population.genome_list.remove(worst_genome)


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
