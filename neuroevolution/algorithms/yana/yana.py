import tensorflow as tf
from random import randint

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class YANA(BaseNeuroevolutionAlgorithm):
    """
    ToDo: Test implementation of the the dummy 'Yet Another Neuroevolution Algorithm', which does all required
          Neuroevolution Algorithm tasks in the most basic way to enable testing the framework.
    """
    def __init__(self, encoding, population, config):
        """
        ToDo
        :param encoding:
        :param config:
        :param population:
        """
        self.logger = tf.get_logger()

        self.encoding = encoding
        self.population = population

        # Read in config parameters for neuroevolution algorithm
        self.pop_size = int(config.get('YANA', 'pop_size'))
        self.genome_removal_percentage = float(config.get('YANA', 'genome_removal_percentage'))

    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        for i in range(self.pop_size):
            genome = self.create_genome(i)
            self.population.add_genome(genome)

        self.population.set_initialized()

    def create_genome(self, genome_id):
        """
        ToDo
        :param genome_id:
        :return:
        """
        return self.encoding.create_genome(genome_id)

    def create_new_generation(self):
        """
        ToDo
        :return:
        """
        # Select and mutate population. Recombining population has been left out for the sake of brevity
        num_genomes_to_remove = int(self.genome_removal_percentage * self.pop_size)
        self._select_genomes(num_genomes_to_remove)
        num_genomes_to_add = self.pop_size - num_genomes_to_remove
        self._mutate_genomes(num_genomes_to_add)

    def _select_genomes(self, num_genomes_to_remove):
        """
        ToDo
        :param num_genomes_to_remove:
        :return:
        """
        for _ in range(num_genomes_to_remove):
            worst_genome = self.population.get_worst_genome()
            self.population.remove_genome(worst_genome)
            self.logger.debug("Genome {} with fitness {} deleted".format(worst_genome.get_id(), worst_genome.get_fitness()))

    def _mutate_genomes(self, num_genomes_to_add):
        """
        ToDo
        :param num_genomes_to_add:
        :return:
        """
        return
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
