import ast
import tensorflow as tf
from random import randint

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

        self.population = population

        # Read in config parameters for neuroevolution algorithm
        self.pop_size = int(config.get('YANA','pop_size'))
        self.available_activations = ast.literal_eval(config.get('YANA','available_activations'))

    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        for i in range(self.pop_size):
            genome = self.create_genome(i)
            self.population.add_genome(genome)

        self.population.set_initialized()

    @staticmethod
    def create_genome(i):
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
        num_genomes_to_add_in_mutation = num_genomes_to_remove
        # num_genomes_to_add_in_mutation = int(pop_size - num_genomes_to_remove / 2)
        # num_genomes_to_add_in_recombination = pop_size - num_genomes_to_remove - num_genomes_to_add_in_mutation

        # Select, Recombine and Mutate Population
        self._select_genomes(num_genomes_to_remove)
        self._mutate_genomes(num_genomes_to_add_in_mutation)
        # self._recombine_genomes(num_genomes_to_add_in_recombination)

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
        added_genomes = 0

        while added_genomes != num_genomes_to_add_in_mutation:
            # Choose a random genome as the basis for the new mutated genome
            new_genome = self.population.genome_list[randint(0, len(self.population.genome_list)-1)]

            # Mutate the new genome repeatedly with probability 33%, though at least once
            while True:
                # Decide if to mutate existing structure or add new structure
                if randint(0, 1) == 0:
                    # Add new structure
                    units = 8 * (2 ** randint(0, 4))
                    activation = self.available_activations[randint(0, 4)]
                    index = randint(1, len(new_genome.phenotype.layer_list)-1)
                    new_genome.phenotype.layer_list.insert(index, tf.keras.layers.Dense(units, activation=activation))
                else:
                    # Mutate existing structure
                    index = randint(1, len(new_genome.phenotype.layer_list)-1)
                    # If last mutate activation function:
                    if index == (len(new_genome.phenotype.layer_list)-1) or randint(0, 1) == 0:
                        # mutate activation function
                        units = new_genome.phenotype.layer_list[index].units
                        activation = self.available_activations[randint(0, 4)]
                        new_genome.phenotype.layer_list[index] = tf.keras.layers.Dense(units, activation=activation)
                    else:
                        units = 8 * (2 ** randint(0, 4))
                        activation = new_genome.phenotype.layer_list[index].activation
                        new_genome.phenotype.layer_list[index] = tf.keras.layers.Dense(units, activation=activation)

                if randint(0, 2) == 0:
                    break

            # Add newly generated genome to population
            added_genomes += 1
            self.population.genome_list.append(new_genome)
            self.logger.debug("Added new mutated genome: {}".format(new_genome))

    def _recombine_genomes(self, num_genomes_to_add_in_recombination):
        """
        ToDo
        :param: num_genomes_to_add_in_recombination
        :return:
        """
        pass

    def check_population_extinction(self):
        """
        ToDo
        :return:
        """
        return len(self.population.genome_list) == 0
