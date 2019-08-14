from collections import deque

import tensorflow as tf


class Population:
    def __init__(self, ne_algorithm, config):
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm

        # Read in config parameters for population
        self.pop_size = config.getint('POPULATION', 'pop_size')

        # create genome container that is the actual population and set generation_counter to uninitialized
        self.genomes = deque(maxlen=self.pop_size)
        self.generation_counter = None

    def initialize(self):
        self.logger.info("Initializing population of size {}".format(self.pop_size))
        for _ in range(self.pop_size):
            new_initialized_genome = self.ne_algorithm.create_initial_genome()
            self.genomes.append(new_initialized_genome)
        self.generation_counter = 0

    def evaluate(self, genome_evaluation_function):
        raise NotImplementedError("evaluate() not yet implemented")

    def evolve(self):
        raise NotImplementedError("evolve() not yet implemented")

    def check_extinction(self):
        return len(self.genomes) == 0

    def summary(self):
        raise NotImplementedError("summary() not yet implemented")

    def get_genome(self, i):
        return self.genomes[i]

    def get_best_genome(self):
        return max(self.genomes, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genomes, key=lambda x: x.get_fitness())

    def get_generation_counter(self):
        return self.generation_counter

    def load_population(self):
        raise NotImplementedError("load_population() not yet implemented")

    def save_population(self):
        raise NotImplementedError("save_population() not yet implemented")


'''
class Population:
    def __init__(self, encoding):
        self.logger = tf.get_logger()

        self.encoding = encoding

        self.genome_list = []
        self.generation_counter = None
        self.initialized_flag = False

    def add_genome(self, genome):
        self.genome_list.append(genome)

    def remove_genome(self, genome):
        self.genome_list.remove(genome)

    def get_genome_list(self):
        return self.genome_list

    def get_genome(self, i):
        return self.genome_list[i]

    def get_best_genome(self):
        return max(self.genome_list, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genome_list, key=lambda x: x.get_fitness())

    def set_initialized(self):
        self.generation_counter = 0
        self.initialized_flag = True

    def increment_generation_counter(self):
        self.generation_counter += 1

    def get_generation_counter(self):
        return self.generation_counter

    def check_extinction(self):
        return len(self.genome_list) == 0

    def save_population(self):
        raise NotImplementedError("save_population() not yet implemented")

    def load_population(self):
        raise NotImplementedError("load_population() not yet implemented")
'''
