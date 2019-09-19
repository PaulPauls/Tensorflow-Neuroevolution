import json
from collections import deque
from absl import logging


class Population:
    def __init__(self, ne_algorithm, config):
        self.ne_algorithm = ne_algorithm

        self.initial_pop_size = None
        self.pop_size_fixed = None
        self._read_config_parameters(config)

        self.pop_size = 0
        self.generation_counter = None

        if self.ne_algorithm.uses_speciation():
            self.species_counter = 1
            self.genomes = {self.species_counter: deque()}
        else:
            self.genomes = deque()

    def _read_config_parameters(self, config):
        self.initial_pop_size = config.getint('POPULATION', 'initial_pop_size')
        self.pop_size_fixed = config.getboolean('POPULATION', 'pop_size_fixed')

        logging.debug("Population read from config: initial_pop_size = {}".format(self.initial_pop_size))
        logging.debug("Population read from config: pop_size_fixed = {}".format(self.pop_size_fixed))

    def initialize(self, input_shape, num_output):
        self.ne_algorithm.initialize_population(self, self.initial_pop_size, input_shape, num_output)
        self.generation_counter = 0
        logging.info("Initialized population to size {}".format(self.pop_size))

    def evaluate(self, genome_eval_function):
        logging.info("Evaluating {} genomes from generation {} ...".format(self.pop_size, self.generation_counter))
        for genome in self.genomes:
            if genome.get_fitness() == 0:
                scored_fitness = genome_eval_function(genome)
                genome.set_fitness(scored_fitness)

    def speciate(self):
        if self.ne_algorithm.uses_speciation():
            self.ne_algorithm.speciate_population(self)

    def evolve(self):
        replacement_count = self.ne_algorithm.evolve_population(self)
        self.generation_counter += 1
        logging.info("Evolution of the population from generation {} to generation {} replaced {} genomes."
                     .format(self.generation_counter - 1, self.generation_counter, replacement_count))

    def summary(self):
        best_fitness = self.get_best_genome().get_fitness()
        average_fitness = self.get_average_fitness()
        logging.info("#### GENERATION: {:>4} ## BEST_FITNESS: {:>8} ## AVERAGE_FITNESS: {:>8} ## POP_SIZE: {:>4} ####"
                     .format(self.generation_counter, best_fitness, average_fitness, self.pop_size))
        for i in range(self.pop_size):
            logging.debug(self.genomes[i])

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

    def load_population(self, encoding, load_file_path):
        with open(load_file_path, 'r') as load_file:
            loaded_population = json.load(load_file)
        self.generation_counter = loaded_population['generation_counter']
        self.pop_size = loaded_population['pop_size']

        genome_list = loaded_population['genomes']
        assert not self.pop_size_fixed or len(genome_list) == self.initial_pop_size
        deserialized_genome_list = encoding.deserialize_genome_list(genome_list)
        if self.pop_size_fixed:
            self.genomes = deque(deserialized_genome_list, maxlen=self.initial_pop_size)
        else:
            self.genomes = deque(deserialized_genome_list)

        logging.info("Loaded population of encoding '{}' from file '{}'. Summary of the population:"
                     .format(encoding.__class__.__name__, load_file_path))
        self.summary()

    def save_population(self, save_file_path):
        serialized_population = {
            'generation_counter': self.generation_counter,
            'pop_size': self.pop_size,
            'genomes': [genome.serialize() for genome in self.genomes]
        }
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_population, save_file, indent=4)
        logging.info("Saved population to file '{}'".format(save_file_path))
