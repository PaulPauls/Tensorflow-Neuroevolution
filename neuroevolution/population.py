from collections import deque

import tensorflow as tf


class Population:
    def __init__(self, ne_algorithm, config):
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm

        # Read in config parameters for population
        self.supplied_pop_size = config.getint('POPULATION', 'pop_size')
        self.limited_pop_size = config.getboolean('POPULATION', 'limited_pop_size')
        self.logger.debug("Population read from config: supplied_pop_size = {}".format(self.supplied_pop_size))
        self.logger.debug("Population read from config: limited_pop_size = {}".format(self.limited_pop_size))

        # create flexible pop_size, genome container that is the actual population and set generation_counter to
        # uninitialized
        self.pop_size = self.supplied_pop_size
        self.genomes = deque(maxlen=self.supplied_pop_size) if self.limited_pop_size else deque()
        self.generation_counter = None

    def initialize(self, input_shape, num_output):
        self.logger.info("Initializing population to size {}".format(self.supplied_pop_size))
        for _ in range(self.supplied_pop_size):
            new_initialized_genome = self.ne_algorithm.create_initial_genome(input_shape, num_output)
            self.genomes.append(new_initialized_genome)
        self.generation_counter = 0

    def evaluate(self, genome_evaluation_function):
        # Evaluate each genome that has so far not been evaluated (effectively having a fitness_score of 0)
        self.logger.debug("Evaluating {} genomes in generation {} ...".format(self.pop_size, self.generation_counter))
        for genome in self.genomes:
            if genome.get_fitness() == 0:
                genome_evaluation_function(genome)
                # self.logger.debug('Genome {} scored fitness {}'.format(genome.get_id(), genome.get_fitness()))

    def evolve(self):
        replacement_count = self.ne_algorithm.create_new_generation(self)
        self.pop_size = len(self.genomes)
        self.generation_counter += 1
        self.logger.info("Evolving the population from generation {} to {} replaced {} genomes.".format(
            self.generation_counter-1, self.generation_counter, replacement_count))

    def summary(self):
        best_fitness = self.get_best_genome().get_fitness()
        average_fitness = self.get_average_fitness()
        self.logger.info("#### GENERATION: {} #### BEST_FITNESS: {} #### AVERAGE_FITNESS: {} #### POP_SIZE: {} ####".
                         format(self.generation_counter, best_fitness, average_fitness, self.pop_size))
        for i in range(self.pop_size):
            self.logger.info(self.genomes[i])
        self.logger.info("#"*100 + "\n")

    def check_extinction(self):
        return self.pop_size == 0

    def append_genome(self, genome):
        self.genomes.append(genome)

    def remove_genome(self, genome):
        self.genomes.remove(genome)

    def get_genome(self, i):
        return self.genomes[i]

    def get_best_genome(self):
        return max(self.genomes, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genomes, key=lambda x: x.get_fitness())

    def get_generation_counter(self):
        return self.generation_counter

    def get_pop_size(self):
        return self.pop_size

    def get_average_fitness(self):
        fitness_sum = sum(genome.get_fitness() for genome in self.genomes)
        average_fitness = round(fitness_sum / self.pop_size, 3)
        return average_fitness

    def load_population(self):
        raise NotImplementedError("load_population() not yet implemented")

    def save_population(self):
        raise NotImplementedError("save_population() not yet implemented")
