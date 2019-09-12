from collections import deque
from absl import logging


class Population:
    def __init__(self, ne_algorithm, config):
        self.ne_algorithm = ne_algorithm

        # Declare and read in config parameters for the population
        self.initial_pop_size = None
        self.pop_size_fixed = None
        self._read_config_parameters(config)

        # create flexible pop_size, container for the genome population and set generation_counter to uninitialized
        self.pop_size = self.initial_pop_size
        self.genomes = deque(maxlen=self.initial_pop_size) if self.pop_size_fixed else deque()
        self.generation_counter = None

    def _read_config_parameters(self, config):
        self.initial_pop_size = config.getint('POPULATION', 'initial_pop_size')
        self.pop_size_fixed = config.getboolean('POPULATION', 'pop_size_fixed')

        logging.debug("Population read from config: initial_pop_size = {}".format(self.initial_pop_size))
        logging.debug("Population read from config: pop_size_fixed = {}".format(self.pop_size_fixed))

    def initialize(self, input_shape, num_output):
        logging.info("Initializing population to size {}...".format(self.initial_pop_size))
        for _ in range(self.initial_pop_size):
            new_initialized_genome = self.ne_algorithm.create_initial_genome(input_shape, num_output)
            self.genomes.append(new_initialized_genome)
        self.generation_counter = 0

    def evaluate(self, genome_eval_function):
        logging.info("Evaluating {} genomes from generation {} ...".format(self.pop_size, self.generation_counter))
        for genome in self.genomes:
            if genome.get_fitness() == 0:
                scored_fitness = genome_eval_function(genome)
                genome.set_fitness(scored_fitness)

    def speciate(self):
        raise NotImplementedError()

    def evolve(self):
        replacement_count = self.ne_algorithm.evolve_population(self)
        self.generation_counter += 1
        logging.info("Evolution of the population from generation {} to generation {} replaced {} genomes."
                     .format(self.generation_counter - 1, self.generation_counter, replacement_count))

    def summary(self):
        best_fitness = self.get_best_genome().get_fitness()
        average_fitness = self.get_average_fitness()
        logging.info("#### GENERATION: {} #### BEST_FITNESS: {} #### AVERAGE_FITNESS: {} #### POP_SIZE: {} ####"
                     .format(self.generation_counter, best_fitness, average_fitness, self.pop_size))
        for i in range(self.pop_size):
            logging.info(self.genomes[i])

    def check_extinction(self):
        return self.pop_size == 0

    def append_genome(self, genome):
        self.genomes.append(genome)
        self.pop_size += 1

    def remove_genome(self, genome):
        self.genomes.remove(genome)
        self.pop_size -= 1

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
        raise NotImplementedError()

    def save_population(self):
        raise NotImplementedError()
