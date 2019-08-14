import tensorflow as tf
from random import random, randint

from neuroevolution.algorithms import BaseNeuroevolutionAlgorithm


class YANA(BaseNeuroevolutionAlgorithm):
    """
    Test implementation of the the dummy 'Yet Another Neuroevolution Algorithm', which does all required Neuroevolution
    Algorithm tasks in the most basic way to enable testing the framework.
    """
    def __init__(self, config):
        self.logger = tf.get_logger()

        # Read in config parameters for neuroevolution algorithm
        self.pop_size = config.getint('NE_ALGORITHM', 'pop_size')
        self.genome_removal_prob = config.getfloat('NE_ALGORITHM', 'genome_removal_prob')
        self.genome_mutate_prob = config.getfloat('NE_ALGORITHM', 'genome_mutate_prob')

    def create_initial_population(self):
        raise NotImplementedError("Should implement create_initial_population()")

    def create_new_generation(self):
        raise NotImplementedError("Should implement create_new_generation()")


'''
class YANA(BaseNeuroevolutionAlgorithm):
    """
    Test implementation of the the dummy 'Yet Another Neuroevolution Algorithm', which does all required Neuroevolution
    Algorithm tasks in the most basic way to enable testing the framework.
    """
    def __init__(self, population, config):
        self.logger = tf.get_logger()

        self.population = population

        # Read in config parameters for neuroevolution algorithm
        self.pop_size = int(config.get('NE_ALGORITHM', 'pop_size'))
        self.genome_removal_prob = float(config.get('NE_ALGORITHM', 'genome_removal_prob'))
        self.genome_mutate_prob = float(config.get('NE_ALGORITHM', 'genome_mutate_prob'))

    def create_initial_population(self):
        for _ in range(self.pop_size):
            genome = self.encoding.create_genome()
            self.population.add_genome(genome)

        self.population.set_initialized()

    def create_new_generation(self):
        # Select and mutate population. Recombining population has been left out for the sake of brevity
        num_genomes_to_remove = int(self.genome_removal_prob * self.pop_size)
        self._select_genomes(num_genomes_to_remove)
        num_genomes_to_add = self.pop_size - num_genomes_to_remove
        self._mutate_genomes(num_genomes_to_add)

    def _select_genomes(self, num_genomes_to_remove):
        for _ in range(num_genomes_to_remove):
            worst_genome = self.population.get_worst_genome()
            self.population.remove_genome(worst_genome)
            self.logger.debug("Genome {} with fitness {} deleted".format(worst_genome.get_id(),
                                                                         worst_genome.get_fitness()))

    def _mutate_genomes(self, num_genomes_to_add):
        added_genomes = 0

        while added_genomes != num_genomes_to_add:
            # Choose a random genome as the basis for the new mutated genome
            new_genome = self.population.get_genome_list()[randint(0, len(self.population.get_genome_list())-1)]
            # Change ID of genome to signify as new
            new_genome.set_id(self.encoding.pop_id())
            new_genome.set_fitness(0)

            # Mutate the new genome repeatedly with specified probability, though at least once
            while True:
                # Decide if to mutate existing structure or add new structure
                if randint(0, 1) == 0:
                    # Add new structure
                    index = randint(1, new_genome.get_layer_count()-1)
                    units = 8 * (2 ** randint(0, 4))
                    activation = self.available_activations[randint(0, 4)]
                    new_genome.add_layer(index, tf.keras.layers.Dense(units, activation=activation))
                else:
                    # Mutate existing structure. Choose which layer
                    index = randint(1, new_genome.get_layer_count()-1)
                    # Check if last layer chosen or chance has it that only activation function is mutated:
                    if index == new_genome.get_layer_count()-1 or randint(0, 1) == 0:
                        # mutate activation function
                        activation = self.available_activations[randint(0, 4)]
                        new_genome.replace_layer(index, tf.keras.layers.Dense, activation=activation)
                    else:
                        # mutate units in layer
                        units = 8 * (2 ** randint(0, 4))
                        new_genome.replace_layer(index, tf.keras.layers.Dense, units=units)

                if random() > self.genome_mutate_prob:
                    break

            # Add newly generated genome to population
            added_genomes += 1
            self.population.add_genome(new_genome)
            self.logger.debug("Added new mutated genome with ID {}".format(new_genome.get_id()))
'''
