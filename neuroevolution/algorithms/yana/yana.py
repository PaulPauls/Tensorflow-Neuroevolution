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

    def create_new_generation(self):
        """
        ToDo
        :return:
        """
        # Determine number of genomes to remove and how many to add through recombination and mutation
        pop_size = int(self.config.algorithm_parameters['YANA']['pop_size'])
        num_genomes_to_remove = int(pop_size * 0.2)
        num_genomes_to_add_in_mutation = int(pop_size - num_genomes_to_remove/2)
        num_genomes_to_add_in_recombination = pop_size - num_genomes_to_remove - num_genomes_to_add_in_mutation

        # Select, Recombine and Mutate Population
        self._select_genomes(num_genomes_to_remove)
        self._mutate_genomes(num_genomes_to_add_in_mutation)
        self._recombine_genomes(num_genomes_to_add_in_recombination)

    def _select_genomes(self, num_genomes_to_remove):
        """
        ToDo
        :param: num_genomes_to_remove
        :return:
        """
        # for now, delete the 20% of the population that is performing the lowest
        for _ in range(num_genomes_to_remove):
            worst_genome = min(self.population.genome_list, key=lambda x: x.fitness)
            self.logger.debug("Genome with fitness {} deleted".format(worst_genome.fitness))
            self.population.genome_list.remove(worst_genome)

    def _mutate_genomes(self, num_genomes_to_add_in_mutation):
        """
        ToDo
        :param: num_genomes_to_add_in_mutation
        :return:
        """

        pass

    def _recombine_genomes(self, num_genomes_to_add_in_recombination):
        """
        ToDo
        :param: num_genomes_to_add_in_recombination
        :return:
        """
        pass
